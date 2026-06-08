import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import customtkinter as ctk  # 核心替换
import pandas as pd
import numpy as np
import os
import datetime
import threading
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import traceback
import warnings

# 保持你原有的科学计算库引用不变...
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score,
                             confusion_matrix, classification_report, roc_curve, auc, log_loss)
from sklearn.inspection import permutation_importance, partial_dependence, PartialDependenceDisplay
from sklearn.impute import SimpleImputer
from scipy import stats

# 原有的 try-except 块保持不变...
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from docx import Document
    from docx.shared import Inches
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import linear_rainbow
    STATS_AVAILABLE = False # 暂时设为False或者保留你的导入逻辑
except ImportError:
    STATS_AVAILABLE = False

# ==========================================
# 全局绘图风格设置
# ==========================================
sns.set_theme(style="whitegrid", font='SimHei')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore")

# ==========================================
# iOS 风格设置
# ==========================================
ctk.set_appearance_mode("Light")  # 强制亮色 (iPhone 白)
ctk.set_default_color_theme("blue")  # 蓝色主题


class ANN_Visualizer:
    """
    负责所有图表绘制 (Matplotlib, Seaborn, Plotly)
    [Update] 新增 SHAP 交互网络图 (包含因变量节点)
    """

    @staticmethod
    def draw_network_topology(model, input_names, target_name, save_dir):
        """绘制静态神经网络拓扑结构图 (PNG)"""
        try:
            coefs = model.coefs_
            n_layers = len(coefs)
            layer_sizes = [len(input_names)] + list(model.hidden_layer_sizes) + [1]

            fig, ax = plt.subplots(figsize=(12, 8))
            ax.axis('off')
            left, right, bottom, top = 0.1, 0.9, 0.1, 0.9
            v_spacing = (top - bottom) / float(max(layer_sizes))
            h_spacing = (right - left) / float(len(layer_sizes) - 1)

            node_coords = []
            for l, n_nodes in enumerate(layer_sizes):
                layer_nodes = []
                layer_top = (top + bottom + (n_nodes - 1) * v_spacing) / 2.0
                for n in range(n_nodes):
                    x = left + l * h_spacing;
                    y = layer_top - n * v_spacing
                    layer_nodes.append((x, y))
                node_coords.append(layer_nodes)

            for l in range(n_layers):
                weights = coefs[l]
                max_wt = np.max(np.abs(weights)) if np.max(np.abs(weights)) > 0 else 1
                for i in range(weights.shape[0]):
                    for j in range(weights.shape[1]):
                        w = weights[i, j]
                        start = node_coords[l][i];
                        end = node_coords[l + 1][j]
                        color = '#1f77b4' if w > 0 else '#b0c4de'
                        alpha = 0.2 + 0.8 * (abs(w) / max_wt)
                        ax.plot([start[0], end[0]], [start[1], end[1]], c=color, linewidth=0.5 + 3 * (abs(w) / max_wt),
                                alpha=alpha, zorder=1)

            box_props = dict(boxstyle="round,pad=0.4", fc="white", ec="black", alpha=0.9)
            circle_props = dict(boxstyle="circle,pad=0.3", fc="#e0e0e0", ec="black", alpha=1.0)

            for l, nodes in enumerate(node_coords):
                for n, (x, y) in enumerate(nodes):
                    if l == 0:
                        lbl = input_names[n];
                        ax.text(x, y, lbl, ha="center", va="center", bbox=box_props, zorder=10, fontsize=9)
                    elif l == len(node_coords) - 1:
                        lbl = target_name;
                        ax.text(x, y, lbl, ha="center", va="center", bbox=box_props, zorder=10, fontsize=9)
                    else:
                        lbl = f"H{l}:{n + 1}";
                        ax.text(x, y, lbl, ha="center", va="center", bbox=circle_props, zorder=10, fontsize=7)

            plt.title(f"ANN Topology: {len(input_names)} Inputs -> {target_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "Fig3_Network_Architecture.png"), dpi=300)
            plt.close()
        except Exception as e:
            print(f"Topology PNG Error: {e}")

    @staticmethod
    def draw_shap_dependence_plot(shap_values, X, feature_names, top_n, save_dir):
        """
        绘制 SHAP 依赖图 (PNG)
        X 应该是原始的 X_test 缩放后的数据
        shap_values 应该是对应的 SHAP 值
        """
        if not SHAP_AVAILABLE: return
        try:
            # 计算平均绝对 SHAP 值并排序，选取 Top N
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            sorted_idx = np.argsort(mean_abs_shap)[::-1][:top_n]

            # 准备数据导出
            dependence_data = []

            for i in sorted_idx:
                feature_idx = i
                feature_name = feature_names[feature_idx]

                # 绘制单个特征的依赖图
                plt.figure(figsize=(10, 8))

                # [修正 1] 兼容性处理：尝试获取交互项索引
                try:
                    # 新版本路径
                    interact_idx = shap.approximate_interactions(feature_idx, shap_values, X)[0]
                except AttributeError:
                    # 旧版本兼容 (防止极老版本报错)
                    try:
                        interact_idx = shap.common.approximate_interactions(feature_idx, shap_values, X)[0]
                    except:
                        interact_idx = "auto"

                # [修正 2] 绘图调用：改回 interaction_index 并传入索引
                shap.dependence_plot(
                    feature_idx,
                    shap_values,
                    X,
                    feature_names=feature_names,
                    display_features=X,
                    interaction_index=interact_idx,  # <--- 关键修改：改回 interaction_index
                    show=False
                )

                plt.title(f"SHAP Dependence Plot: {feature_name}")
                plt.tight_layout()
                # 使用 600 DPI 导出
                plt.savefig(os.path.join(save_dir, f"Fig_SHAP_Dependence_{feature_name}.png"), dpi=600,
                            bbox_inches='tight')
                plt.close()

                # 提取绘图数据 (用于 XLSX)
                feat_vals = X[:, feature_idx]
                shap_vals = shap_values[:, feature_idx]

                # 获取交互项的数据 (如果 interact_idx 是数字)
                if isinstance(interact_idx, int):
                    interact_name = feature_names[interact_idx]
                    interact_vals = X[:, interact_idx]
                else:
                    interact_name = "Auto/None"
                    interact_vals = [0] * len(feat_vals)

                # 记录数据
                dependence_data.extend([
                    {'Feature': feature_name,
                     'Feature Value (Scaled)': feat_vals[k],
                     'SHAP Value': shap_vals[k],
                     'Interaction Feature': interact_name,
                     'Interaction Value': interact_vals[k] if k < len(interact_vals) else 0
                     } for k in range(len(feat_vals))
                ])

            df_dep = pd.DataFrame(dependence_data)
            df_dep.to_excel(os.path.join(save_dir, "Table_SHAP_Dependence_Data.xlsx"), index=False)

        except Exception as e:
            print(f"SHAP Dependence Plot Error: {e}")
            import traceback
            traceback.print_exc()

    @staticmethod
    def draw_shap_advanced_plots(shap_values, X, feature_names, base_value, save_dir):
        """
        绘制高级 SHAP 图表 (Beeswarm, Bar, Waterfall, Force, Decision)
        并导出对应的 PNG (600 DPI) 和 XLSX 数据
        """
        if not SHAP_AVAILABLE: return

        try:
            # ==========================================================
            # [关键修复] 构造 shap.Explanation 对象
            # 必须确保 base_values 是数组格式，否则会导致 'float' has no attribute 'shape' 错误
            # ==========================================================

            # 1. 确保 shap_values 是 numpy 数组
            if isinstance(shap_values, list):
                shap_values = np.array(shap_values)

            # 2. 确保 base_value 被扩展为数组 (N_samples,)
            if np.isscalar(base_value):
                # 如果是单个数，则复制 N 次，N 为样本数 (X 的行数)
                base_values_expanded = np.full(X.shape[0], base_value)
            else:
                # 如果已经是数组，直接使用
                base_values_expanded = base_value

            # 3. 创建 Explanation 对象
            shap_exp = shap.Explanation(
                values=shap_values,
                base_values=base_values_expanded,
                data=X,
                feature_names=feature_names
            )

            # ==========================================================
            # 1. 蜂群图 (Beeswarm Plot)
            # ==========================================================
            try:
                plt.figure(figsize=(10, 8))
                shap.plots.beeswarm(shap_exp, show=False, max_display=20)
                plt.title("SHAP Beeswarm Plot", fontsize=16)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, "Fig_SHAP_1_Beeswarm.png"), dpi=600, bbox_inches='tight')
                plt.close()

                # 数据导出
                df_beeswarm = pd.DataFrame(X, columns=[f"Feat_{f}" for f in feature_names])
                df_shap_vals = pd.DataFrame(shap_values, columns=[f"SHAP_{f}" for f in feature_names])
                df_full = pd.concat([df_beeswarm, df_shap_vals], axis=1)
                df_full.to_excel(os.path.join(save_dir, "Table_SHAP_1_Beeswarm_Data.xlsx"), index=False)
            except Exception as e:
                print(f"SHAP Beeswarm Error: {e}")

            # ==========================================================
            # 2. 条形图 (Bar Plot)
            # ==========================================================
            try:
                plt.figure(figsize=(10, 8))
                shap.plots.bar(shap_exp, show=False, max_display=20)

                # 数据标签逻辑略复杂，静态图导出即可
                plt.title("SHAP Global Importance (Bar Plot)", fontsize=16)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, "Fig_SHAP_2_Bar.png"), dpi=600, bbox_inches='tight')
                plt.close()

                # 数据导出
                mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
                sorted_idx = np.argsort(mean_abs_shap)[::-1]
                df_bar = pd.DataFrame({
                    "Feature": np.array(feature_names)[sorted_idx],
                    "Mean_Abs_SHAP": mean_abs_shap[sorted_idx]
                })
                df_bar.to_excel(os.path.join(save_dir, "Table_SHAP_2_Bar_Data.xlsx"), index=False)
            except Exception as e:
                print(f"SHAP Bar Plot Error: {e}")
                import traceback
                traceback.print_exc()

            # ==========================================================
            # 3. 瀑布图 (Waterfall Plot)
            # ==========================================================
            try:
                sample_idx = 0
                plt.figure(figsize=(10, 8))
                # 传入单个样本的 Explanation 对象
                shap.plots.waterfall(shap_exp[sample_idx], show=False, max_display=15)
                plt.title(f"SHAP Waterfall Plot (Sample {sample_idx})", fontsize=16)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, "Fig_SHAP_3_Waterfall.png"), dpi=600, bbox_inches='tight')
                plt.close()

                # 数据导出
                # 获取该样本对应的基准值 (如果是数组取第0个，如果是标量直接取)
                current_base = base_values_expanded[sample_idx] if isinstance(base_values_expanded,
                                                                              np.ndarray) else base_values_expanded

                df_waterfall = pd.DataFrame({
                    "Feature": feature_names,
                    "Feature_Value": X[sample_idx],
                    "SHAP_Value": shap_values[sample_idx]
                })
                df_waterfall['Abs_SHAP'] = df_waterfall['SHAP_Value'].abs()
                df_waterfall = df_waterfall.sort_values(by='Abs_SHAP', ascending=False).drop(columns=['Abs_SHAP'])
                df_waterfall.loc['Base_Value'] = ["BASE VALUE", "N/A", current_base]
                df_waterfall.to_excel(os.path.join(save_dir, "Table_SHAP_3_Waterfall_Data.xlsx"), index=False)
            except Exception as e:
                print(f"SHAP Waterfall Error: {e}")

            # ==========================================================
            # 4. 力图 (Force Plot)
            # ==========================================================
            try:
                sample_idx = 0
                plt.figure(figsize=(20, 4))  # 宽一点

                # Force plot 比较特殊，它倾向于使用原始数值
                current_base = base_values_expanded[sample_idx] if isinstance(base_values_expanded,
                                                                              np.ndarray) else base_values_expanded

                shap.force_plot(
                    current_base,
                    shap_values[sample_idx],
                    X[sample_idx],
                    feature_names=feature_names,
                    matplotlib=True,
                    show=False
                )
                # Force plot 的 matplotlib 模式通常不需要 tight_layout，因为它是硬编码布局
                plt.savefig(os.path.join(save_dir, "Fig_SHAP_4_Force.png"), dpi=600, bbox_inches='tight')
                plt.close()

                # 数据导出 (复用 Waterfall 数据结构)
                df_waterfall.to_excel(os.path.join(save_dir, "Table_SHAP_4_Force_Data.xlsx"), index=False)
            except Exception as e:
                print(f"SHAP Force Plot Error: {e}")

            # ==========================================================
            # 5. 决策图 (Decision Plot)
            # ==========================================================
            try:
                n_display = min(20, len(X))
                # 决策图同样需要 base_value
                current_base = base_values_expanded[0]  # Decision plot通常假设统一基准值，或者传入列表

                plt.figure(figsize=(10, 8))
                shap.decision_plot(
                    current_base,
                    shap_values[:n_display],
                    X[:n_display],
                    feature_names=feature_names,
                    show=False
                )
                plt.title(f"SHAP Decision Plot (Top {n_display} Samples)", fontsize=16)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, "Fig_SHAP_5_Decision.png"), dpi=600, bbox_inches='tight')
                plt.close()

                # 数据导出
                decision_data = []
                for i in range(n_display):
                    row_data = {"Sample_Index": i, "Base_Value": current_base}
                    cumulative = current_base
                    for feat, val in zip(feature_names, shap_values[i]):
                        row_data[f"SHAP_{feat}"] = val
                        cumulative += val
                    row_data["Final_Prediction"] = cumulative
                    decision_data.append(row_data)

                df_decision = pd.DataFrame(decision_data)
                df_decision.to_excel(os.path.join(save_dir, "Table_SHAP_5_Decision_Data.xlsx"), index=False)
            except Exception as e:
                print(f"SHAP Decision Plot Error: {e}")

        except Exception as main_e:
            print(f"SHAP Advanced Plots Main Error: {main_e}")
            import traceback
            traceback.print_exc()






    @staticmethod
    def export_shap_force_plot_data(shap_values, X, feature_names, explainer, save_dir):
        """
        力瀑布图 (Force Plot) 的数据导出 (静态图难以高质量导出，故只导出数据)
        导出行数最多 50 行
        """
        if not SHAP_AVAILABLE: return
        try:
            # 选取前 50 行数据进行代表性展示 (或其他逻辑)
            n_rows = min(len(X), 50)

            # 基础值 (Base Value)
            base_value = explainer.expected_value

            # 准备数据
            force_plot_data = []

            for i in range(n_rows):
                shap_row = shap_values[i, :]
                X_row = X[i, :]

                # 每一行代表一个观测值的 Force Plot 数据
                data_point = {
                    'Observation_Index': i,
                    'Base Value (E[f(x)])': base_value,
                    'Model Output (f(x))': base_value + np.sum(shap_row)  # 这是预测值
                }

                # 添加每个特征的 SHAP 值和特征值
                for name, s_val, x_val in zip(feature_names, shap_row, X_row):
                    data_point[f'SHAP_{name}'] = s_val
                    data_point[f'Feature_{name} (Scaled)'] = x_val

                force_plot_data.append(data_point)

            df_force = pd.DataFrame(force_plot_data)
            df_force.to_excel(os.path.join(save_dir, "Table_SHAP_Force_Data.xlsx"), index=False)

        except Exception as e:
            print(f"SHAP Force Plot Data Error: {e}")





    @staticmethod
    def draw_network_topology_html(model, input_names, target_name, save_dir):
        """[New] 绘制交互式神经网络拓扑结构图 (HTML)"""
        if not PLOTLY_AVAILABLE: return
        try:
            coefs = model.coefs_
            layer_sizes = [len(input_names)] + list(model.hidden_layer_sizes) + [1]
            n_layers = len(layer_sizes)

            x_coords = []
            y_coords = []
            node_text = []
            node_color = []

            for l_idx, size in enumerate(layer_sizes):
                x_layer = l_idx
                y_step = 1.0 / (size + 1)
                for n_idx in range(size):
                    x_coords.append(x_layer)
                    y_pos = 1.0 - (n_idx + 1) * y_step
                    y_coords.append(y_pos)

                    if l_idx == 0:
                        txt = f"Input: {input_names[n_idx]}"
                        col = '#1f77b4'
                    elif l_idx == n_layers - 1:
                        txt = f"Output: {target_name}"
                        col = '#d62728'
                    else:
                        txt = f"Hidden Layer {l_idx}<br>Neuron {n_idx + 1}"
                        col = '#ff7f0e'
                    node_text.append(txt)
                    node_color.append(col)

            fig = go.Figure()

            curr_idx = 0
            layer_indices = []
            for size in layer_sizes:
                layer_indices.append(list(range(curr_idx, curr_idx + size)))
                curr_idx += size

            for l in range(len(coefs)):
                weights = coefs[l]
                src_indices = layer_indices[l]
                dst_indices = layer_indices[l + 1]
                max_wt = np.max(np.abs(weights)) if np.max(np.abs(weights)) > 0 else 1

                for i in range(weights.shape[0]):
                    for j in range(weights.shape[1]):
                        w = weights[i, j]
                        if abs(w) < 0.01: continue

                        src_id = src_indices[i]
                        dst_id = dst_indices[j]
                        line_color = 'rgba(31, 119, 180, 0.6)' if w > 0 else 'rgba(176, 196, 222, 0.6)'
                        width = 0.5 + 4 * (abs(w) / max_wt)

                        fig.add_trace(go.Scatter(
                            x=[x_coords[src_id], x_coords[dst_id], None],
                            y=[y_coords[src_id], y_coords[dst_id], None],
                            mode='lines',
                            line=dict(width=width, color=line_color),
                            hoverinfo='none',
                            showlegend=False,
                            opacity=0.8
                        ))

            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='markers+text',
                marker=dict(size=20, color=node_color, line=dict(width=2, color='white')),
                text=[t.split(':')[1][:5] if ':' in t else '' for t in node_text],
                textposition="top center",
                hovertext=node_text,
                hoverinfo="text",
                name="Neurons"
            ))

            fig.update_layout(
                title="Interactive Network Topology (Zoom/Hover enabled)",
                showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white',
                dragmode='pan'
            )
            fig.write_html(os.path.join(save_dir, "Interactive_Network_Architecture.html"))
        except Exception as e:
            print(f"Topology HTML Error: {e}")

    @staticmethod
    def generate_pdp_html(model, X, features, save_dir):
        """生成 PDP 交互式 HTML"""
        if not PLOTLY_AVAILABLE: return
        try:
            fig = go.Figure()
            for i, name in enumerate(features[:6]):
                pdp_res = partial_dependence(model, X, [i])
                fig.add_trace(go.Scatter(x=pdp_res['grid_values'][0], y=pdp_res['average'][0], mode='lines', name=name))
            fig.update_layout(title="Interactive Partial Dependence Plot (PDP)", xaxis_title="Variable Value (Scaled)",
                              yaxis_title="Predicted Impact")
            fig.write_html(os.path.join(save_dir, "Interactive_PDP.html"))
        except Exception as e:
            print(f"PDP Error: {e}")

    @staticmethod
    def generate_3d_html(model, inputs, top2, target, save_dir):
        """生成 3D 响应曲面 HTML"""
        if not PLOTLY_AVAILABLE: return
        try:
            idx1, idx2 = inputs.index(top2[0]), inputs.index(top2[1])
            x1 = np.linspace(0, 1, 50);
            x2 = np.linspace(0, 1, 50)
            X1, X2 = np.meshgrid(x1, x2)
            mesh = np.zeros((2500, len(inputs))) + 0.5
            mesh[:, idx1] = X1.ravel();
            mesh[:, idx2] = X2.ravel()
            Z = model.predict(mesh).reshape(X1.shape)
            fig = go.Figure(data=[go.Surface(z=Z, x=X1, y=X2)])
            fig.update_layout(title=f"3D: {top2[0]} & {top2[1]} -> {target}",
                              scene=dict(xaxis_title=top2[0], yaxis_title=top2[1], zaxis_title=target))
            fig.write_html(os.path.join(save_dir, "Interactive_3D_Surface.html"))
        except:
            pass

    @staticmethod
    def generate_shap_html(shap_values, X, feature_names, save_dir, explainer=None):
        """生成 SHAP 相关的交互式图表 (Summary + Force)"""
        if not PLOTLY_AVAILABLE: return

        try:
            # 1. Summary Plot (Beeswarm style)
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            sorted_idx = np.argsort(mean_abs_shap)
            sorted_features = [feature_names[i] for i in sorted_idx]

            fig = go.Figure()
            for idx in sorted_idx:
                feat_name = feature_names[idx]
                s_vals = shap_values[:, idx]
                f_vals = X[:, idx]

                fig.add_trace(go.Scatter(
                    x=s_vals,
                    y=[feat_name] * len(s_vals),
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=f_vals,
                        colorscale='RdBu',
                        showscale=True,
                        reversescale=True,
                        colorbar=dict(title="Feature Value") if idx == sorted_idx[-1] else None
                    ),
                    name=feat_name,
                    text=[f"Val: {v:.2f}<br>Shap: {s:.2f}" for v, s in zip(f_vals, s_vals)],
                    hoverinfo='text'
                ))

            fig.update_layout(
                title="Interactive SHAP Summary Plot",
                xaxis_title="SHAP Value (Impact)",
                yaxis_title="Features",
                showlegend=False,
                height=max(400, len(feature_names) * 40)
            )
            fig.write_html(os.path.join(save_dir, "Interactive_SHAP_Summary.html"))

            # 2. Force Plot (Individual)
            if SHAP_AVAILABLE and explainer is not None:
                try:
                    import shap
                    p = shap.force_plot(explainer.expected_value, shap_values[0, :], X[0, :],
                                        feature_names=feature_names, show=False)
                    shap.save_html(os.path.join(save_dir, "Interactive_SHAP_Force_Individual.html"), p)
                except:
                    pass

        except Exception as e:
            print(f"SHAP HTML Gen Error: {e}")

    @staticmethod
    def draw_shap_correlation_network_html(shap_values, feature_names, save_dir, y_values=None, target_name="Target"):
        """
        [New] 生成 SHAP 交互/关联网络图 (Circular Layout)
        **特别增强**: 将因变量 (Target) 作为节点加入网络。
        """
        if not PLOTLY_AVAILABLE: return
        try:
            # 1. 基础计算
            # 节点大小：平均绝对 SHAP 值 (Global Importance)
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

            # --- 构建扩展的矩阵以包含 Target ---
            n_features = len(feature_names)
            # 归一化节点大小
            max_imp = np.max(mean_abs_shap) if np.max(mean_abs_shap) > 0 else 1
            node_sizes = (mean_abs_shap / max_imp) * 40 + 10

            # 2. 布局计算 (圆形布局 Circular Layout)
            # 我们将 n_features 个特征分布在圆周上，Target 放在圆心或者也放在圆周上？
            # 为了美观，特征在圆周，Target 在圆心

            radius = 1.0
            angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False)
            x_coords = list(radius * np.cos(angles))
            y_coords = list(radius * np.sin(angles))

            # 添加 Target 节点 (在圆心)
            x_coords.append(0.0)
            y_coords.append(0.0)
            node_sizes = list(node_sizes) + [60]  # Target 节点大一点

            all_names = feature_names + [target_name]
            all_colors = list(mean_abs_shap) + [max_imp * 1.2]  # Target 颜色最深

            # 3. 计算相关性 (Edges)
            df_shap = pd.DataFrame(shap_values, columns=feature_names)
            corr_matrix = df_shap.corr().values

            fig = go.Figure()

            threshold = 0.2  # 阈值，过滤弱连接
            max_width = 8

            # A. 绘制特征之间的连接 (圆周上的连线)
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    corr = corr_matrix[i, j]
                    if abs(corr) > threshold:
                        edge_color = f"rgba(255, 0, 0, {abs(corr) * 0.8})" if corr > 0 else f"rgba(0, 0, 255, {abs(corr) * 0.8})"
                        width = abs(corr) * max_width

                        fig.add_trace(go.Scatter(
                            x=[x_coords[i], x_coords[j], None],
                            y=[y_coords[i], y_coords[j], None],
                            mode='lines',
                            line=dict(width=width, color=edge_color),
                            hoverinfo='none', opacity=0.7, showlegend=False
                        ))

            # B. 绘制特征与 Target 的连接 (辐射状连线)
            # 逻辑：计算 特征SHAP值 与 真实Y值 的相关性
            if y_values is not None:
                # 确保 y_values 是 1D array
                y_flat = np.array(y_values).ravel()
                target_idx = n_features  # Target 在列表最后

                for i in range(n_features):
                    # 计算 corr(Shap_Feature_i, Y_True)
                    # 这代表该特征对结果的"驱动一致性"
                    try:
                        feat_shap = shap_values[:, i]
                        # 避免常数数组导致的 NaN
                        if np.std(feat_shap) > 1e-9 and np.std(y_flat) > 1e-9:
                            r_val = np.corrcoef(feat_shap, y_flat)[0, 1]
                        else:
                            r_val = 0

                        if abs(r_val) > 0.1:  # 稍微低一点的阈值
                            edge_color = f"rgba(0, 128, 0, {abs(r_val)})" if r_val > 0 else f"rgba(255, 165, 0, {abs(r_val)})"
                            # Green for pos, Orange for neg (distinct from Feat-Feat)
                            width = abs(r_val) * (max_width + 2)

                            fig.add_trace(go.Scatter(
                                x=[x_coords[i], x_coords[target_idx], None],
                                y=[y_coords[i], y_coords[target_idx], None],
                                mode='lines',
                                line=dict(width=width, color=edge_color, dash='solid'),
                                hoverinfo='text',
                                hovertext=f"{feature_names[i]} -> Target<br>Corr: {r_val:.2f}",
                                showlegend=False
                            ))
                    except:
                        pass

            # 4. 绘制节点 (Nodes)
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='markers+text',
                marker=dict(
                    size=node_sizes,
                    color=all_colors,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Importance")
                ),
                text=[n[:5] for n in all_names],
                textposition="top center",
                hovertext=all_names,
                hoverinfo="text",
                name="Nodes"
            ))

            fig.update_layout(
                title=f"SHAP Network: Features & {target_name}",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.5, 1.5]),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.5, 1.5]),
                plot_bgcolor='white',
                width=900, height=900,
                showlegend=False,
                annotations=[dict(text="Center: Target<br>Perimeter: Features<br>Green/Orange Lines: Impact on Target",
                                  x=-1.4, y=-1.4, showarrow=False, align='left')]
            )

            fig.write_html(os.path.join(save_dir, "Interactive_SHAP_Network.html"))
        except Exception as e:
            print(f"SHAP Network Error: {e}")

    @staticmethod
    def plot_residuals(y_true, y_pred, save_dir, html_dir):
        """生成残差分析图 (PNG + HTML)"""
        residuals = y_true - y_pred
        std_resid = residuals / (np.std(residuals) + 1e-9)

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.scatter(y_pred, std_resid, alpha=0.5, c='blue', edgecolors='k')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values');
        plt.ylabel('Standardized Residuals')
        plt.title('1. Heteroscedasticity Check')

        plt.subplot(1, 3, 2)
        sns.histplot(residuals, kde=True, color='green')
        plt.title('2. Residual Distribution')

        plt.subplot(1, 3, 3)
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('3. Q-Q Plot')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "Fig_Residual_Analysis.png"), dpi=300)
        plt.close()

        if PLOTLY_AVAILABLE:
            fig1 = px.scatter(x=y_pred, y=std_resid, title="Interactive: Residuals vs Predicted",
                              labels={'x': 'Predicted Value', 'y': 'Std Residuals'})
            fig1.add_hline(y=0, line_dash="dash", line_color="red")
            fig1.write_html(os.path.join(html_dir, "Interactive_Residuals_Scatter.html"))

            fig2 = px.histogram(residuals, nbins=30, title="Interactive: Residual Distribution",
                                labels={'value': 'Residual Error'})
            fig2.write_html(os.path.join(html_dir, "Interactive_Residuals_Hist.html"))

    @staticmethod
    def plot_classification_metrics(y_true, y_prob, y_pred, save_dir_fig, save_dir_html, save_dir_rep):
        """生成分类图表: Confusion Matrix, ROC, Lift"""
        y_true = y_true.astype(int)
        cm = confusion_matrix(y_true, y_pred)

        cm_df = pd.DataFrame(cm, index=[f'Actual_{i}' for i in range(cm.shape[0])],
                             columns=[f'Pred_{i}' for i in range(cm.shape[1])])
        cm_df.to_excel(os.path.join(save_dir_rep, "Table_Confusion_Matrix.xlsx"))

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix');
        plt.ylabel('Actual');
        plt.xlabel('Predicted')
        plt.savefig(os.path.join(save_dir_fig, "Fig_Confusion_Matrix.png"), dpi=300)
        plt.close()

        if PLOTLY_AVAILABLE:
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                               title="Interactive Confusion Matrix")
            fig_cm.write_html(os.path.join(save_dir_html, "Interactive_Confusion_Matrix.html"))

        if len(np.unique(y_true)) == 2:
            pos_probs = y_prob[:, 1]
            fpr, tpr, _ = roc_curve(y_true, pos_probs)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(6, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
            plt.plot([0, 1], [0, 1], 'navy', linestyle='--')
            plt.title('ROC Curve');
            plt.legend()
            plt.savefig(os.path.join(save_dir_fig, "Fig_ROC_Curve.png"), dpi=300)
            plt.close()

            if PLOTLY_AVAILABLE:
                fig_roc = px.area(x=fpr, y=tpr, title=f'ROC Curve (AUC={roc_auc:.4f})',
                                  labels=dict(x='False Positive Rate', y='True Positive Rate'))
                fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                fig_roc.write_html(os.path.join(save_dir_html, "Interactive_ROC_Curve.html"))

            data = pd.DataFrame({'y_true': y_true, 'y_prob': pos_probs})
            data = data.sort_values(by='y_prob', ascending=False)
            data['decile'] = pd.qcut(data['y_prob'].rank(method='first'), 10, labels=False)
            data['decile'] = 9 - data['decile'] + 1
            lift_df = data.groupby('decile').agg(Count=('y_true', 'count'), Responders=('y_true', 'sum')).reset_index()

            avg_resp = lift_df['Responders'].sum() / lift_df['Count'].sum()
            lift_df['Lift'] = (lift_df['Responders'] / lift_df['Count']) / avg_resp
            lift_df['Cum_Gain'] = lift_df['Responders'].cumsum() / lift_df['Responders'].sum()
            lift_df.to_excel(os.path.join(save_dir_rep, "Table_Marketing_Lift_Gain.xlsx"), index=False)

            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.bar(lift_df['decile'], lift_df['Lift'], color='#b3cde0', label='Lift')
            ax1.set_ylabel('Lift Value', color='blue');
            ax1.axhline(y=1, color='red', linestyle='--')
            ax2 = ax1.twinx()
            ax2.plot(lift_df['decile'], lift_df['Cum_Gain'], color='#d62728', marker='o', label='Cum. Gain')
            ax2.set_ylabel('Cumulative Gain (%)', color='#d62728')
            plt.title('Decile Lift & Cumulative Gain')
            plt.savefig(os.path.join(save_dir_fig, "Fig_Lift_Gain_Chart.png"), dpi=300)
            plt.close()

            if PLOTLY_AVAILABLE:
                fig_lg = go.Figure()
                fig_lg.add_trace(go.Bar(x=lift_df['decile'], y=lift_df['Lift'], name="Lift"))
                fig_lg.add_trace(go.Scatter(x=lift_df['decile'], y=lift_df['Cum_Gain'], name="Cum Gain", yaxis='y2'))
                fig_lg.update_layout(title="Interactive Lift & Gain", yaxis=dict(title="Lift"),
                                     yaxis2=dict(title="Gain", overlaying='y', side='right'))
                fig_lg.write_html(os.path.join(save_dir_html, "Interactive_Lift_Gain.html"))


class ANN_Analytics:
    """
    负责数学计算与逻辑分析 (Elasticity, Robustness)
    [Fixed] 修复了分类模式下稳健性检验报错的问题
    """

    @staticmethod
    def calculate_elasticity(model, X_scaled, scaler_X, scaler_y, inputs, save_dir, html_dir, log_func=None):
        if log_func: log_func("正在计算营销弹性 (Elasticity)...")
        elasticity_results = {}

        X_orig = scaler_X.inverse_transform(X_scaled)
        y_pred_sc = model.predict(X_scaled)
        y_orig = scaler_y.inverse_transform(y_pred_sc.reshape(-1, 1)).ravel()
        y_base = y_orig.copy()
        y_base[y_base == 0] = 1e-6  # Avoid division by zero

        delta = 0.01  # 1% change

        for i, col in enumerate(inputs):
            X_plus = X_orig.copy()
            X_plus[:, i] = X_plus[:, i] * (1 + delta)
            X_plus_sc = scaler_X.transform(X_plus)
            y_new_sc = model.predict(X_plus_sc)
            y_new = scaler_y.inverse_transform(y_new_sc.reshape(-1, 1)).ravel()

            pct_change_y = (y_new - y_orig) / y_base
            elasticity_results[col] = np.mean(pct_change_y / delta)

        df_e = pd.DataFrame(list(elasticity_results.items()), columns=['Feature', 'Mean Elasticity'])
        df_e = df_e.sort_values(by='Mean Elasticity', ascending=False)
        df_e.to_excel(os.path.join(save_dir, "Table_Marketing_Elasticity.xlsx"), index=False)

        if PLOTLY_AVAILABLE:
            fig = px.bar(df_e, x='Mean Elasticity', y='Feature', orientation='h',
                         title="Marketing Elasticity Analysis (1% Change Impact)",
                         color='Mean Elasticity', text_auto='.3f')
            fig.write_html(os.path.join(html_dir, "Interactive_Elasticity.html"))

        if log_func: log_func("✅ 弹性分析完成: Excel + HTML")
        return df_e

    @staticmethod
    def run_robustness_check(df, inputs, target, scaler_X, scaler_y, base_model_params, save_dir, log_func=None,
                             mode='Regression', le=None):
        if log_func: log_func(f"正在执行稳健性检验 ({mode} Mode)...")
        robust_res = []

        # 1. 根据模式选择模型类和处理逻辑
        if mode == 'Classification':
            from sklearn.neural_network import MLPClassifier
            ModelClass = MLPClassifier
        else:
            from sklearn.neural_network import MLPRegressor
            ModelClass = MLPRegressor

        for k in range(5):
            # 采样 90%
            df_sub = df.sample(frac=0.9, random_state=k * 100)
            X_sub = scaler_X.transform(df_sub[inputs].values)

            # 2. 根据模式处理 Y (LabelEncoder vs Scaler)
            if mode == 'Classification':
                # 分类模式：使用 LabelEncoder 转换
                y_sub = le.transform(df_sub[target].values)
            else:
                # 回归模式：使用 MinMaxScaler 转换
                y_sub = scaler_y.transform(df_sub[target].values.reshape(-1, 1)).ravel()

            # 3. 训练与计算重要性
            model = ModelClass(**base_model_params)
            model.fit(X_sub, y_sub)

            perm = permutation_importance(model, X_sub, y_sub, n_repeats=3, random_state=42)
            res = {"Run": f"Run_{k + 1}"}
            for i, col in enumerate(inputs):
                res[col] = perm.importances_mean[i]
            robust_res.append(res)

        df_rob = pd.DataFrame(robust_res)
        df_mean = df_rob.mean(numeric_only=True)
        df_std = df_rob.std(numeric_only=True)
        df_summary = pd.DataFrame({"Mean_Importance": df_mean, "Std_Dev": df_std, "CV": df_std / (df_mean + 1e-9)})
        df_summary.to_excel(os.path.join(save_dir, "Table_Robustness_Check.xlsx"))
        if log_func: log_func("✅ 稳健性检验完成")


class ANN_Reporter:
    """
    负责生成 Word 报告
    """

    @staticmethod
    def generate_word_report(save_dir, inputs, target, t8, df_t9, fig_dir, linearity_result):
        if not DOCX_AVAILABLE: return
        doc = Document()
        doc.add_heading(f'ANN Analysis Report: {target}', 0)
        doc.add_paragraph(f"Generated Date: {datetime.datetime.now()}")

        # 1. Linearity
        doc.add_heading('1. Linearity Test', level=1)
        if linearity_result:
            p_val = linearity_result['P-Value']
            conclusion = "significant non-linearity (p < 0.05)" if p_val < 0.05 else "no significant non-linearity"
            doc.add_paragraph(f"Ramsey RESET p-value: {p_val:.5f}. Result: {conclusion}.")
        else:
            doc.add_paragraph("Linearity test not performed.")

        # 2. Performance
        doc.add_heading('2. Model Performance', level=1)
        avg_r2 = np.mean([x['R2_Testing'] for x in t8]) if 'R2_Testing' in t8[0] else 0
        doc.add_paragraph(f"Average Testing R2: {avg_r2:.3f}")

        # 3. Importance
        doc.add_heading('3. Variable Importance', level=1)
        top_var = df_t9.loc['Normalized'].drop('Network').idxmax()
        doc.add_paragraph(f"Most influential predictor: {top_var}")
        if os.path.exists(os.path.join(fig_dir, "Fig_Sensitivity.png")):
            doc.add_picture(os.path.join(fig_dir, "Fig_Sensitivity.png"), width=Inches(5.5))

        # 4. Topology
        doc.add_heading('4. Architecture', level=1)
        if os.path.exists(os.path.join(fig_dir, "Fig3_Network_Architecture.png")):
            doc.add_picture(os.path.join(fig_dir, "Fig3_Network_Architecture.png"), width=Inches(5.5))

        doc.save(os.path.join(save_dir, "Full_Analysis_Report.docx"))


class SemAnnProApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("PLS-ANN Analytical Tool V1.8 By JW❤QX @小红书drharry")
        self.geometry("1400x950")

        # 兼容旧代码引用
        self.root = self

        # --- 核心变量 ---
        self.df = None
        self.output_dir = ""
        self.debug_mode = False
        self.linearity_result = None

        # --- 界面构建 ---
        self._init_ui()

    def _init_ui(self):
        # 网格布局：左侧导航(0)，右侧内容(1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # === 1. 左侧侧边栏 ===
        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")
        self.sidebar.grid_rowconfigure(5, weight=1)  # 让底部空出来

        # 标题
        ctk.CTkLabel(self.sidebar, text="PLS-ANN Pro", font=("Arial", 22, "bold")).grid(row=0, column=0, padx=20,
                                                                                        pady=(20, 10))

        # 导航按钮
        self.btn_analysis = self._create_nav_btn("1. 基础建模", "analysis", 1)
        self.btn_advanced = self._create_nav_btn("2. 高级配置", "advanced", 2)
        self.btn_predict = self._create_nav_btn("3. 模型应用", "predict", 3)

        # 侧边栏底部日志
        ctk.CTkLabel(self.sidebar, text="System Log", font=("Arial", 12, "bold"), text_color="gray").grid(row=6,
                                                                                                          column=0,
                                                                                                          padx=20,
                                                                                                          sticky="w")
        self.txt_log = ctk.CTkTextbox(self.sidebar, height=150, font=("Consolas", 10))
        self.txt_log.grid(row=7, column=0, padx=10, pady=10, sticky="ew")
        self.txt_log.configure(state="disabled")

        # === 2. 右侧主内容区 ===
        self.main_area = ctk.CTkFrame(self, fg_color="transparent")
        self.main_area.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_area.grid_rowconfigure(0, weight=1)
        self.main_area.grid_columnconfigure(0, weight=1)

        # 创建页面容器
        self.frames = {}
        for name in ["analysis", "advanced", "predict"]:
            frame = ctk.CTkFrame(self.main_area, fg_color="white", corner_radius=15)  # 白色卡片
            frame.grid(row=0, column=0, sticky="nsew")
            self.frames[name] = frame

        # 初始化各页面
        self.setup_analysis_ui(self.frames["analysis"])
        self.setup_advanced_ui(self.frames["advanced"])
        self.setup_predict_ui(self.frames["predict"])

        # 默认显示第一页
        self.select_frame("analysis")

    def _create_nav_btn(self, text, name, row):
        btn = ctk.CTkButton(self.sidebar, corner_radius=0, height=50, border_spacing=10, text=text,
                            fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                            anchor="w", command=lambda n=name: self.select_frame(n))
        btn.grid(row=row, column=0, sticky="ew")
        return btn

    def select_frame(self, name):
        # 按钮高亮逻辑
        btns = {"analysis": self.btn_analysis, "advanced": self.btn_advanced, "predict": self.btn_predict}
        for n, btn in btns.items():
            btn.configure(fg_color=("gray75", "gray25") if n == name else "transparent")

        # 切换页面
        self.frames[name].tkraise()

        # 特殊逻辑：如果是切到 advanced，同步变量
        if name == "advanced":
            self.sync_variables()

    def sync_variables(self):
        # 从 Basic Tab 同步变量到 Advanced Tab 的列表
        self.list_inputs_adv.delete(0, tk.END)
        self.list_target_adv.delete(0, tk.END)

        input_idxs = self.list_inputs_basic.curselection()
        for i in input_idxs:
            self.list_inputs_adv.insert(tk.END, self.list_inputs_basic.get(i))

        target_idxs = self.list_target_basic.curselection()
        for i in target_idxs:
            self.list_target_adv.insert(tk.END, self.list_target_basic.get(i))

    def create_tabs(self):
        self.notebook = ttk.Notebook(self.root, bootstyle="primary")  # 增加样式
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)

        # Tab 1: 基础建模
        self.tab_analysis = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_analysis, text=" 1. 基础建模 (Modeling) ")
        self.setup_analysis_ui()

        # Tab 2: 高级配置
        self.tab_advanced = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_advanced, text=" 2. 高级配置 (Advanced) ")
        self.setup_advanced_ui()

        # Tab 3: 模型应用
        self.tab_predict = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_predict, text=" 3. 模型应用 (Predict) ")
        self.setup_predict_ui()

        # --- 新增：绑定标签切换事件，用于同步变量显示 ---
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)

        # 底部日志区 - 模仿终端风格
        frame_log = ttk.Labelframe(self.root, text=" System Log / 运行日志 ", padding=5, bootstyle="secondary")
        frame_log.pack(fill="both", expand=True, padx=10, pady=5)

        # 黑色背景，绿色文字，Consolas 字体
        self.txt_log = tk.Text(frame_log, height=10, bg="#0f0f0f", fg="#00ff00",
                               font=("Consolas", 9), insertbackground="white", relief="flat")
        self.txt_log.pack(fill="both", expand=True)

    def on_tab_change(self, event):
        """
        当切换标签页时触发。
        如果是切换到 Tab 2 (Advanced)，则自动从 Tab 1 同步已选的变量进行显示。
        """
        # 获取当前选中的 Tab 索引
        try:
            selected_tab_index = self.notebook.index(self.notebook.select())
        except:
            return

        # 如果选中的是 Tab 2 (索引为 1)
        if selected_tab_index == 1:
            # 1. 清空 Tab 2 的显示列表
            self.list_inputs_adv.delete(0, tk.END)
            self.list_target_adv.delete(0, tk.END)

            # 2. 同步自变量 (Inputs)
            # 获取 Tab 1 中被选中的索引
            input_idxs = self.list_inputs_basic.curselection()
            for i in input_idxs:
                # 获取文本并插入 Tab 2
                val = self.list_inputs_basic.get(i)
                self.list_inputs_adv.insert(tk.END, val)

            # 3. 同步因变量 (Target)
            target_idxs = self.list_target_basic.curselection()
            for i in target_idxs:
                val = self.list_target_basic.get(i)
                self.list_target_adv.insert(tk.END, val)



    # ============================================================
    # UI: Tab 1 基础建模
    # ============================================================
    def setup_analysis_ui(self, parent):
        parent.grid_columnconfigure(0, weight=1)

        # 标题
        ctk.CTkLabel(parent, text="基础建模 (Modeling)", font=("Arial", 20, "bold"), text_color="#333").pack(anchor="w",
                                                                                                             padx=20,
                                                                                                             pady=20)

        # === A. 数据源配置 ===
        f_data = ctk.CTkFrame(parent, fg_color="#F5F5F7", corner_radius=10)
        f_data.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(f_data, text="A. 数据源", font=("Arial", 14, "bold"), text_color="#007AFF").grid(row=0, column=0,
                                                                                                      sticky="w",
                                                                                                      padx=15, pady=10)

        ctk.CTkButton(f_data, text="📂 导入 CSV/Excel", command=self.load_data, fg_color="#007AFF").grid(row=1, column=0,
                                                                                                        padx=15, pady=5)
        self.lbl_file = ctk.CTkLabel(f_data, text="未加载数据", text_color="gray")
        self.lbl_file.grid(row=1, column=1, padx=10, sticky="w")

        self.var_debug = tk.BooleanVar(value=False)
        ctk.CTkSwitch(f_data, text="Debug 模式", variable=self.var_debug).grid(row=1, column=3, padx=20)

        ctk.CTkButton(f_data, text="📂 输出目录", command=self.select_output_dir, fg_color="gray").grid(row=2, column=0,
                                                                                                       padx=15, pady=5)
        self.lbl_dir = ctk.CTkLabel(f_data, text="默认: 当前目录", text_color="gray")
        self.lbl_dir.grid(row=2, column=1, padx=10, sticky="w")

        ctk.CTkLabel(f_data, text="📢 提示: 样本量建议至少为自变量数量的 50 倍", text_color="#FF9500",
                     font=("Arial", 10)).grid(row=3, column=0, columnspan=4, sticky="w", padx=15, pady=(0, 10))

        # === B. 分析模式 ===
        f_mode = ctk.CTkFrame(parent, fg_color="#F5F5F7", corner_radius=10)
        f_mode.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(f_mode, text="B. 分析模式", font=("Arial", 14, "bold"), text_color="#007AFF").pack(anchor="w",
                                                                                                        padx=15,
                                                                                                        pady=10)
        self.var_mode = tk.StringVar(value="Regression")

        # 使用 SegmentedButton (iOS 风格的分段控制器)
        self.seg_mode = ctk.CTkSegmentedButton(f_mode, values=["Regression (数值预测)", "Classification (类别分类)"],
                                               variable=self.var_mode, dynamic_resizing=True)
        self.seg_mode.pack(fill="x", padx=20, pady=(0, 20))

        # === C. 变量映射 ===
        f_vars = ctk.CTkFrame(parent, fg_color="transparent")
        f_vars.pack(fill="both", expand=True, padx=20, pady=10)

        # 左侧 Inputs
        c_left = ctk.CTkFrame(f_vars, fg_color="#F5F5F7")
        c_left.pack(side="left", fill="both", expand=True, padx=(0, 10))
        ctk.CTkLabel(c_left, text="自变量 (Inputs / X)", font=("Arial", 12, "bold")).pack(pady=5)

        lb_frame1 = ctk.CTkFrame(c_left)
        lb_frame1.pack(fill="both", expand=True, padx=5, pady=5)

        # [修改关键点]: 添加 exportselection=False
        self.list_inputs_basic = tk.Listbox(lb_frame1, selectmode="multiple", height=8, borderwidth=0,
                                            highlightthickness=0, font=("Arial", 11), exportselection=False)
        self.list_inputs_basic.pack(fill="both", expand=True, padx=2, pady=2)

        # 右侧 Target
        c_right = ctk.CTkFrame(f_vars, fg_color="#F5F5F7")
        c_right.pack(side="right", fill="both", expand=True, padx=(10, 0))
        ctk.CTkLabel(c_right, text="因变量 (Target / Y)", font=("Arial", 12, "bold"), text_color="#FF3B30").pack(pady=5)

        lb_frame2 = ctk.CTkFrame(c_right)
        lb_frame2.pack(fill="both", expand=True, padx=5, pady=5)

        # [修改关键点]: 添加 exportselection=False
        self.list_target_basic = tk.Listbox(lb_frame2, selectmode="single", height=8, borderwidth=0,
                                            highlightthickness=0, font=("Arial", 11), exportselection=False)
        self.list_target_basic.pack(fill="both", expand=True, padx=2, pady=2)

        # === D. 基础参数 ===
        f_param = ctk.CTkFrame(parent, fg_color="#F5F5F7")
        f_param.pack(fill="x", padx=20, pady=10)

        g = ctk.CTkFrame(f_param, fg_color="transparent")
        g.pack(pady=10)

        ctk.CTkLabel(g, text="隐藏层数:").grid(row=0, column=0, padx=5)
        self.ent_layer_num = ctk.CTkEntry(g, width=60);
        self.ent_layer_num.insert(0, "2");
        self.ent_layer_num.grid(row=0, column=1)

        ctk.CTkLabel(g, text="神经元策略:").grid(row=0, column=2, padx=5)
        self.cb_neuron_rule = ctk.CTkOptionMenu(g, values=["Input Size (N)-经验最稳", "Kolmogorov (2N+1)-理论最强", "Mean ((N+O)/2)-分析最高",
                                                           "Custom（自定义）"], width=180, command=self.toggle_custom_neuron)
        self.cb_neuron_rule.grid(row=0, column=3, padx=5)

        self.ent_neurons_manual = ctk.CTkEntry(g, width=80, placeholder_text="Custom");
        self.ent_neurons_manual.grid(row=0, column=4, padx=5)
        self.ent_neurons_manual.configure(state="disabled")

        ctk.CTkLabel(g, text="激活函数:").grid(row=1, column=0, padx=5, pady=10)
        self.cb_activation = ctk.CTkOptionMenu(g, values=["Sigmoid(logistic)", "tanh", "relu", "identity"]);
        self.cb_activation.grid(row=1, column=1, columnspan=2, sticky="ew")

        ctk.CTkLabel(g, text="折数 (K):").grid(row=1, column=3, padx=5)
        self.ent_folds = ctk.CTkEntry(g, width=60);
        self.ent_folds.insert(0, "10");
        self.ent_folds.grid(row=1, column=4)

        # 训练集占比
        ctk.CTkLabel(g, text="训练集占比(%):").grid(row=2, column=0, padx=5)
        self.var_train_pct = tk.StringVar(value="90")
        self.var_train_pct.trace_add("write", self.update_test_pct)
        self.ent_split = ctk.CTkEntry(g, width=60, textvariable=self.var_train_pct);
        self.ent_split.grid(row=2, column=1)

        ctk.CTkLabel(g, text="测试集(%):").grid(row=2, column=2, padx=5)
        self.var_test_pct = tk.StringVar(value="10")
        self.ent_test_split = ctk.CTkEntry(g, width=60, textvariable=self.var_test_pct, state="disabled");
        self.ent_test_split.grid(row=2, column=3)

        # 运行按钮
        ctk.CTkButton(parent, text="🚀 开始全流程分析 (RUN ANALYSIS)", height=50, font=("Arial", 16, "bold"),
                      fg_color="#34C759", command=self.start_thread).pack(fill="x", padx=40, pady=20)
    def toggle_custom_neuron(self, choice):
        if "Custom" in choice:
            self.ent_neurons_manual.configure(state="normal")
        else:
            self.ent_neurons_manual.configure(state="disabled")

    def update_test_pct(self, *args):
        try:
            val = float(self.var_train_pct.get())
            if 0 <= val <= 100: self.var_test_pct.set(f"{100 - val:.0f}")
        except:
            pass
    # ============================================================
    # UI: Tab 2 高级配置
    # ============================================================
    def setup_advanced_ui(self, parent):
        ctk.CTkLabel(parent, text="高级配置 (Advanced)", font=("Arial", 20, "bold"), text_color="#333").pack(anchor="w",
                                                                                                             padx=20,
                                                                                                             pady=20)

        # 初始化变量
        self.var_autotune = tk.BooleanVar(value=False);
        self.var_earlystop = tk.BooleanVar(value=True)
        self.var_impute = tk.BooleanVar(value=False);
        self.var_outlier = tk.BooleanVar(value=False)
        self.var_linearity = tk.BooleanVar(value=False)
        self.var_shap = tk.BooleanVar(value=False);
        self.var_pdp = tk.BooleanVar(value=False)
        self.var_3d = tk.BooleanVar(value=False);
        self.var_save_model = tk.BooleanVar(value=True)
        self.var_resid = tk.BooleanVar(value=True);
        self.var_elasticity = tk.BooleanVar(value=True)
        self.var_robust = tk.BooleanVar(value=False)

        # A. 算法优化
        f_opt = ctk.CTkFrame(parent, fg_color="#F5F5F7")
        f_opt.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(f_opt, text="A. 算法优化与清洗", font=("Arial", 14, "bold"), text_color="#34C759").pack(anchor="w",
                                                                                                             padx=15,
                                                                                                             pady=10)

        g1 = ctk.CTkFrame(f_opt, fg_color="transparent")
        g1.pack(fill="x", padx=15, pady=5)
        ctk.CTkSwitch(g1, text="自动超参数寻优", variable=self.var_autotune).grid(row=0, column=0, padx=20, pady=10,
                                                                                  sticky="w")
        ctk.CTkSwitch(g1, text="早停机制 (Early Stopping)", variable=self.var_earlystop).grid(row=0, column=1, padx=20,
                                                                                              pady=10, sticky="w")
        ctk.CTkSwitch(g1, text="缺失值填补 (Mean)", variable=self.var_impute).grid(row=1, column=0, padx=20, pady=10,
                                                                                   sticky="w")
        ctk.CTkSwitch(g1, text="异常值剔除 (Z>3)", variable=self.var_outlier).grid(row=1, column=1, padx=20, pady=10,
                                                                                   sticky="w")
        ctk.CTkSwitch(g1, text="非线性检验 (Ramsey)", variable=self.var_linearity).grid(row=1, column=2, padx=20,
                                                                                        pady=10, sticky="w")

        # B. 可视化
        f_viz = ctk.CTkFrame(parent, fg_color="#F5F5F7")
        f_viz.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(f_viz, text="B. 高级可视化", font=("Arial", 14, "bold"), text_color="#FF3B30").pack(anchor="w",
                                                                                                         padx=15,
                                                                                                         pady=10)

        g2 = ctk.CTkFrame(f_viz, fg_color="transparent")
        g2.pack(fill="x", padx=15, pady=5)

        switches = [
            (self.var_shap, "SHAP 可解释性", 0, 0), (self.var_pdp, "偏依赖图 (PDP)", 0, 1),
            (self.var_3d, "3D 响应曲面", 0, 2),
            (self.var_save_model, "保存模型 (.pkl)", 1, 0), (self.var_resid, "残差分析", 1, 1),
            (self.var_elasticity, "营销弹性分析", 1, 2),
            (self.var_robust, "稳健性检验", 2, 0)
        ]
        for var, txt, r, c in switches:
            ctk.CTkSwitch(g2, text=txt, variable=var).grid(row=r, column=c, padx=20, pady=10, sticky="w")

        # C. 变量确认 (只读列表)
        f_vars = ctk.CTkFrame(parent, fg_color="transparent")
        f_vars.pack(fill="x", padx=20, pady=10)

        c1 = ctk.CTkFrame(f_vars, fg_color="#F5F5F7")
        c1.pack(side="left", fill="both", expand=True, padx=(0, 10))
        ctk.CTkLabel(c1, text="已选自变量 (Inputs)").pack()
        self.list_inputs_adv = tk.Listbox(c1, height=5, borderwidth=0, bg="#F5F5F7");
        self.list_inputs_adv.pack(fill="both", padx=5, pady=5)

        c2 = ctk.CTkFrame(f_vars, fg_color="#F5F5F7")
        c2.pack(side="right", fill="both", expand=True, padx=(10, 0))
        ctk.CTkLabel(c2, text="已选因变量 (Target)").pack()
        self.list_target_adv = tk.Listbox(c2, height=5, borderwidth=0, bg="#F5F5F7");
        self.list_target_adv.pack(fill="both", padx=5, pady=5)

        ctk.CTkButton(parent, text="🚀 开始全流程分析 (RUN ANALYSIS)", height=50, font=("Arial", 16, "bold"),
                      fg_color="#34C759", command=self.start_thread).pack(fill="x", padx=40, pady=20)

    # ============================================================
    # UI: Tab 3 模型应用
    # ============================================================
    def setup_predict_ui(self, parent):
        ctk.CTkLabel(parent, text="模型应用 & 模拟器", font=("Arial", 20, "bold"), text_color="#333").pack(anchor="w",
                                                                                                           padx=20,
                                                                                                           pady=20)

        # 1. 批量预测
        f_batch = ctk.CTkFrame(parent, fg_color="#F5F5F7")
        f_batch.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(f_batch, text="1. 批量文件预测", font=("Arial", 14, "bold"), text_color="#007AFF").pack(anchor="w",
                                                                                                             padx=15,
                                                                                                             pady=10)

        g = ctk.CTkFrame(f_batch, fg_color="transparent")
        g.pack(fill="x", padx=15, pady=5)

        ctk.CTkButton(g, text="📂 加载模型 (.pkl)", command=self.load_trained_model, fg_color="gray").grid(row=0,
                                                                                                          column=0,
                                                                                                          padx=10,
                                                                                                          pady=5)
        self.lbl_model_status = ctk.CTkLabel(g, text="未加载", text_color="gray");
        self.lbl_model_status.grid(row=0, column=1, sticky="w")

        ctk.CTkButton(g, text="📄 加载数据 (.csv)", command=self.load_predict_data, fg_color="gray").grid(row=1,
                                                                                                         column=0,
                                                                                                         padx=10,
                                                                                                         pady=5)
        self.lbl_predict_file = ctk.CTkLabel(g, text="未加载", text_color="gray");
        self.lbl_predict_file.grid(row=1, column=1, sticky="w")

        ctk.CTkButton(f_batch, text="▶ 运行预测并导出 Excel", command=self.run_prediction, fg_color="#007AFF").pack(
            fill="x", padx=40, pady=20)

        # 2. 模拟器
        f_sim = ctk.CTkFrame(parent, fg_color="#E8F5E9")  # 淡绿色背景
        f_sim.pack(fill="both", expand=True, padx=20, pady=10)
        ctk.CTkLabel(f_sim, text="2. 营销策略模拟器 (Marketing Simulator)", font=("Arial", 14, "bold"),
                     text_color="#2E7D32").pack(anchor="w", padx=15, pady=10)

        ctk.CTkLabel(f_sim,
                     text="加载模型后，启动模拟器进行 What-If 分析。\n通过拖动滑块改变自变量数值，实时查看预测结果。",
                     text_color="gray").pack(pady=10)

        ctk.CTkButton(f_sim, text="🎮 启动交互式模拟器", height=60, font=("Arial", 16, "bold"), fg_color="#34C759",
                      command=self.open_simulator).pack(pady=30)

    def open_simulator(self):
        if not hasattr(self, 'loaded_pack'):
            messagebox.showwarning("Warn", "请先在上方步骤1中加载已训练的模型 (.pkl)")
            return

        sim_win = ctk.CTkToplevel(self)
        sim_win.title("Strategy Simulator")
        sim_win.geometry("800x900")

        model = self.loaded_pack['model']
        sc_x = self.loaded_pack['sc_x']
        sc_y = self.loaded_pack['sc_y']
        cols = self.loaded_pack['cols']

        # 顶部结果
        f_res = ctk.CTkFrame(sim_win, fg_color="#333", corner_radius=0)
        f_res.pack(fill="x")
        ctk.CTkLabel(f_res, text="PREDICTED OUTCOME", text_color="gray").pack(pady=(20, 5))
        self.lbl_sim_res = ctk.CTkLabel(f_res, text="0.00", font=("Arial", 60, "bold"), text_color="#34C759")
        self.lbl_sim_res.pack(pady=(0, 20))

        # 滚动区域
        scroll = ctk.CTkScrollableFrame(sim_win, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=20, pady=20)

        self.sliders = {}
        min_vals = sc_x.data_min_
        max_vals = sc_x.data_max_

        # [Fix] 增加默认参数 val=None，防止手动调用不传参时报错
        def update_prediction(val=None):
            row = [self.sliders[c].get() for c in cols]
            X_trans = sc_x.transform([row])
            y_out_sc = model.predict(X_trans)
            y_out = sc_y.inverse_transform(y_out_sc.reshape(-1, 1)).ravel()[0]
            self.lbl_sim_res.configure(text=f"{y_out:.4f}")

        for i, col in enumerate(cols):
            f = ctk.CTkFrame(scroll, fg_color="#F5F5F7")
            f.pack(fill="x", pady=5)

            vmin, vmax = float(min_vals[i]), float(max_vals[i])
            if vmin == vmax: vmax += 1.0

            ctk.CTkLabel(f, text=f"{col}", font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=(5, 0))

            s = ctk.CTkSlider(f, from_=vmin, to=vmax, command=update_prediction)
            s.set((vmin + vmax) / 2)
            s.pack(fill="x", padx=10, pady=10)

            self.sliders[col] = s

            f_lbl = ctk.CTkFrame(f, fg_color="transparent")
            f_lbl.pack(fill="x", padx=10, pady=(0, 5))
            ctk.CTkLabel(f_lbl, text=f"{vmin:.2f}", font=("Arial", 10)).pack(side="left")
            ctk.CTkLabel(f_lbl, text=f"{vmax:.2f}", font=("Arial", 10)).pack(side="right")

        # [Fix] 触发一次初始预测，传入0防止报错
        update_prediction(0)
    # ============================================================
    # UI: Tab 4 使用说明书与阈值
    # ============================================================
    def setup_help_ui(self):
        # 使用 ScrolledText 配合深色背景
        txt_help = scrolledtext.ScrolledText(self.tab_help, width=100, height=30,
                                             font=("Consolas", 10), bg="#2b3e50", fg="white", relief="flat")
        txt_help.pack(fill="both", expand=True, padx=10, pady=10)

        help_content = """
【Universal ANN Researcher Tool v6.0 使用说明书】

一、 指标阈值与判定标准 (Thresholds & Criteria)
--------------------------------------------------
1. R² (决定系数 / Coefficient of Determination):
   - 判定模型解释力的大小。范围：0 - 1
   - 阈值 (Marketing Context):
     * R² < 0.25: 微弱 (Weak)
     * 0.25 ≤ R² < 0.50: 中等 (Moderate)
     * R² ≥ 0.75: 显著/强 (Substantial)

2. RMSE (均方根误差 / Root Mean Square Error):
   - 判定预测的精准度。越小越好。
   - 若 RMSE_Training 远小于 RMSE_Testing，则存在过拟合 (Overfitting)。

3. P-value (Ramsey RESET 非线性检验):
   - 判定数据是否适合用 ANN。
   - 阈值：
     * P < 0.05: 显著。拒绝线性假设 -> **强烈推荐使用 ANN**。
     * P ≥ 0.05: 不显著。线性模型可能已足够。

二、 输出文件含义 (Output Files)
--------------------------------------------------
1. [Table_Linearity_Test.xlsx]: **Ramsey RESET 检验结果 (F值, P值)**。
2. [Table8_Metrics.xlsx]: 每一折的 RMSE 和 R²，以及平均值和标准差。
3. [Table9_Importance.xlsx]: 变量重要性得分。
4. [Full_Analysis_Report.docx]: **中英双语对照**的完整分析报告。
5. [Interactive_PDP.html]: **偏依赖图的交互式网页版**。

三、 图表含义 (Charts)
--------------------------------------------------
1. Fig_Sensitivity.png: 变量重要性排序。
2. Fig_PDP.png / Interactive_PDP.html: 展示非线性关系曲线。
3. Fig_SHAP.png: 样本级的正负影响分析。
4. Fig3_Network.png: 神经网络拓扑结构。
   -线条的颜色通常代表权重的正负（例如：代码中蓝色代表正权重，浅灰色/浅蓝色代表负权重）。
   -线条越粗：代表该连接的权重绝对值越大，说明两个神经元之间的联系越紧密、影响力越强。
   -线条越细：代表该连接的权重越接近 0，说明影响力较弱。

四、 常见问题
--------------------------------------------------
Q: 为什么 Table_Linearity_Test.xlsx 里的 P 值是空的？
A: 请确保勾选了"非线性检验"，且数据量足够进行 OLS 回归。
        """
        txt_help.insert(tk.END, help_content)
        txt_help.config(state="disabled")

    # ============================================================
    # 核心逻辑
    # ============================================================
    def log(self, msg, is_debug=False):
        if is_debug and not self.var_debug.get(): return
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        prefix = "[DEBUG] " if is_debug else ""

        self.txt_log.configure(state="normal")
        self.txt_log.insert("end", f"[{ts}] {prefix}{msg}\n")
        self.txt_log.see("end")
        self.txt_log.configure(state="disabled")

    def load_data(self):
        path = filedialog.askopenfilename(filetypes=[("Data Files", "*.csv *.xlsx")])
        if path:
            try:
                if path.endswith(".csv"):
                    self.df = pd.read_csv(path)
                else:
                    self.df = pd.read_excel(path)

                self.lbl_file.configure(text=os.path.basename(path))  # fix: configure
                self.log(f"数据加载成功: {len(self.df)} 行")

                self.list_inputs_basic.delete(0, tk.END)
                self.list_target_basic.delete(0, tk.END)

                for col in self.df.columns:
                    if pd.api.types.is_numeric_dtype(self.df[col]):
                        self.list_inputs_basic.insert(tk.END, col)
                        self.list_target_basic.insert(tk.END, col)
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def select_output_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.output_dir = path
            self.lbl_dir.configure(text=path)  # fix: configure

    def load_trained_model(self):
        p = filedialog.askopenfilename(filetypes=[("Model", "*.pkl")])
        if p:
            self.loaded_pack = joblib.load(p)
            # [Fix] CustomTkinter 必须用 .configure，不能用 .config
            # [Fix] bootstyle 是无效参数，改为 text_color="green"
            self.lbl_model_status.configure(text="模型已加载", text_color="green")

    def load_predict_data(self):
        p = filedialog.askopenfilename()
        if p:
            if p.endswith(".csv"):
                self.pred_df = pd.read_csv(p)
            else:
                self.pred_df = pd.read_excel(p)
            # [Fix] CustomTkinter 必须用 .configure，不能用 .config
            self.lbl_predict_file.configure(text=f"数据加载: {len(self.pred_df)}行")

    def start_thread(self):
        if self.df is None: messagebox.showwarning("Warn", "请先加载数据"); return
        threading.Thread(target=self.safe_run_analysis).start()

    def safe_run_analysis(self):
        try:
            self.run_analysis()
        except Exception as e:
            self.log(f"❌ 错误: {str(e)}")
            if self.var_debug.get():
                self.log(traceback.format_exc(), is_debug=True)
            else:
                self.log("请开启 Debug 查看详情")

    def run_analysis(self):
        # 局部导入必要的评价指标函数，防止顶部未导入报错
        from sklearn.metrics import precision_recall_fscore_support

        raw_mode = self.var_mode.get()
        if "Classification" in raw_mode:
            mode = "Classification"
        else:
            mode = "Regression"

        # 1. 变量获取：从 Tab 1 (list_inputs_basic) 获取
        inputs_idx = self.list_inputs_basic.curselection()
        target_idx = self.list_target_basic.curselection()

        if not inputs_idx or not target_idx:
            self.log("❌ 错误: 未选择变量 (请在 Tab 1 中选择自变量和因变量)")
            messagebox.showwarning("提示", "请在 Tab 1 中选择自变量 (Inputs) 和因变量 (Target)")
            return

        inputs = [self.list_inputs_basic.get(i) for i in inputs_idx]
        target = self.list_target_basic.get(target_idx[0])

        if target in inputs:
            inputs = [col for col in inputs if col != target]
            self.log(f"⚠️ 自动修正: 检测到自变量中包含因变量，已自动移除 '{target}'。")

        self.log(f"🚀 开始分析 [{mode} Mode]... Inputs: {len(inputs)}, Target: {target}")

        # 目录创建
        base_path = self.output_dir if self.output_dir else os.getcwd()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        res_dir = os.path.join(base_path, f"ANN_{mode}_{timestamp}")
        dir_rep = os.path.join(res_dir, "01_Reports")
        dir_fig = os.path.join(res_dir, "02_Figures")
        dir_html = os.path.join(res_dir, "03_HTML_Interactive")
        dir_mod = os.path.join(res_dir, "04_Models")

        for d in [dir_rep, dir_fig, dir_html, dir_mod]:
            os.makedirs(d, exist_ok=True)

        # Debug Log: 系统环境
        if self.var_debug.get():
            self.log(f"OS: {os.name}, PID: {os.getpid()}", is_debug=True)
            self.log(f"Output Directory: {res_dir}", is_debug=True)

        # 2. 预处理
        df_work = self.df.copy()
        needed_cols = inputs + [target]

        # Debug Log: 原始数据
        if self.var_debug.get():
            self.log(f"Raw Data Shape: {df_work.shape}", is_debug=True)
            self.log(f"Missing Values Before: \n{df_work[needed_cols].isnull().sum()}", is_debug=True)

        if self.var_impute.get():
            imp = SimpleImputer(strategy='mean')
            df_work[needed_cols] = imp.fit_transform(df_work[needed_cols])
            self.log("✅ 缺失值填补完成 (Mean)")

        if self.var_outlier.get():
            if mode == 'Regression':
                z = np.abs(stats.zscore(df_work[needed_cols].select_dtypes(include=np.number)))
                old_len = len(df_work)
                df_work = df_work[(z < 3).all(axis=1)]
                self.log(f"✅ 异常值剔除: 移除了 {old_len - len(df_work)} 行 (Z>3)")
            else:
                self.log("⚠️ 提示: 分类模式下已自动跳过'异常值剔除'。")

        scaler_X = MinMaxScaler()
        X = df_work[inputs].values
        y = df_work[target].values
        X_sc = scaler_X.fit_transform(X)

        # Debug Log: 数据变换
        if self.var_debug.get():
            self.log(f"X Scaled Range: Min={X_sc.min():.3f}, Max={X_sc.max():.3f}", is_debug=True)

        # 准备 Y (用于后续计算 Output Size)
        if mode == "Classification":
            le = LabelEncoder()
            y_enc = le.fit_transform(y)
            scaler_y = None
            n_classes = len(np.unique(y_enc))
            n_out = n_classes if n_classes > 2 else 1
            if self.var_debug.get():
                self.log(f"Classes found: {le.classes_}", is_debug=True)
        else:
            scaler_y = MinMaxScaler()
            y_enc = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
            n_out = 1  # 回归任务输出层通常为 1

        # ==========================================================
        # 非线性检验 (Ramsey RESET)
        # ==========================================================
        self.linearity_result = None  # 重置结果
        if self.var_linearity.get():
            if mode == 'Regression':
                self.log("正在执行非线性检验 (Ramsey RESET)...")
                try:
                    import statsmodels.api as sm
                    from statsmodels.stats.diagnostic import linear_reset

                    # 准备数据：添加常数项用于 OLS
                    X_with_const = sm.add_constant(X_sc)
                    # 建立线性回归模型
                    ols_model = sm.OLS(y_enc, X_with_const).fit()

                    # 执行 RESET 检验
                    reset_res = linear_reset(ols_model, power=2, test_type='fitted', use_f=True)

                    # 兼容不同版本的返回值提取
                    p_val = reset_res.pvalue if hasattr(reset_res, 'pvalue') else reset_res[1]
                    f_val = reset_res.fvalue if hasattr(reset_res, 'fvalue') else reset_res[0]

                    # 判定结论
                    conclusion = "Non-Linear (Recommend ANN)" if p_val < 0.05 else "Linear (OLS sufficient)"

                    # 保存 Excel
                    lin_df = pd.DataFrame({
                        "Test Name": ["Ramsey RESET Test"],
                        "F-Statistic": [f_val],
                        "P-Value": [p_val],
                        "Significance Level": [0.05],
                        "Conclusion": [conclusion]
                    })
                    save_path_lin = os.path.join(dir_rep, "Table_Linearity_Test.xlsx")
                    lin_df.to_excel(save_path_lin, index=False)

                    self.linearity_result = {'P-Value': p_val}
                    self.log(f"✅ 非线性检验完成: P={p_val:.4f} (已保存至 01_Reports)")

                except ImportError:
                    self.log("❌ 错误: 未安装 statsmodels 库，无法执行非线性检验。")
                except Exception as e:
                    self.log(f"⚠️ 非线性检验运行失败: {str(e)}")
            else:
                self.log("⚠️ 提示: 非线性检验 (RESET) 仅适用于回归模式，已跳过。")

        # 3. 训练参数解析 (这是默认的手动参数)
        # [修改] 增加健壮的 try-except 块，专门处理 Custom 输入解析
        try:
            train_pct = float(self.var_train_pct.get()) / 100
            n_folds = int(self.ent_folds.get())

            act_selection = self.cb_activation.get()
            if "logistic" in act_selection:
                activation_func = "logistic"
            elif "tanh" in act_selection:
                activation_func = "tanh"
            elif "identity" in act_selection:
                activation_func = "identity"
            else:
                activation_func = "relu"

            n_layers = int(self.ent_layer_num.get())
            neuron_strategy = self.cb_neuron_rule.get()
            n_in = len(inputs)

            if "Input Size" in neuron_strategy:
                neurons_per_layer = n_in
                self.log(f"ℹ️ 神经元策略: Input Size (N) -> 每层 {neurons_per_layer} 个")
                hl_sizes = tuple([neurons_per_layer] * n_layers)
            elif "Kolmogorov" in neuron_strategy:
                neurons_per_layer = 2 * n_in + 1
                self.log(f"ℹ️ 神经元策略: Kolmogorov (2N+1) -> 每层 {neurons_per_layer} 个")
                hl_sizes = tuple([neurons_per_layer] * n_layers)
            elif "Mean" in neuron_strategy:
                neurons_per_layer = (n_in + n_out) // 2
                if neurons_per_layer < 1: neurons_per_layer = 1
                self.log(f"ℹ️ 神经元策略: Mean ((N+O)/2) -> 每层 {neurons_per_layer} 个")
                hl_sizes = tuple([neurons_per_layer] * n_layers)
            elif "Custom" in neuron_strategy:
                # [关键修复] 增强 Custom 模式的解析逻辑
                try:
                    custom_val = self.ent_neurons_manual.get().strip()
                    # 替换中文逗号，移除多余空格
                    custom_val = custom_val.replace("，", ",").replace(" ", "")

                    if not custom_val:
                        raise ValueError("输入为空")

                    if "," in custom_val:
                        # 输入如 "10,20" -> (10, 20)
                        parts = custom_val.split(',')
                        # 确保都是整数
                        hl_sizes = tuple([int(p) for p in parts if p])
                        self.log(f"ℹ️ 神经元策略: Custom (List) -> {hl_sizes}")
                    else:
                        # 输入如 "10" -> (10, 10, ...) 根据层数
                        neurons_per_layer = int(custom_val)
                        if neurons_per_layer <= 0: raise ValueError("神经元数必须正整数")
                        hl_sizes = tuple([neurons_per_layer] * n_layers)
                        self.log(f"ℹ️ 神经元策略: Custom (Fixed) -> 每层 {neurons_per_layer} 个")

                except Exception as e_custom:
                    self.log(f"⚠️ 自定义神经元参数错误: {e_custom}。将自动降级为 'Input Size' 策略。", is_debug=True)
                    messagebox.showwarning("参数错误",
                                           f"自定义神经元格式错误: {self.ent_neurons_manual.get()}\n系统将使用默认策略 (Input Size) 继续运行。")
                    hl_sizes = tuple([n_in] * n_layers)
            else:
                hl_sizes = tuple([n_in] * n_layers)

        except Exception as e:
            self.log(f"⚠️ 全局参数解析错误，将使用安全默认值: {e}")
            n_folds = 10
            train_pct = 0.9
            hl_sizes = (len(inputs), len(inputs))
            activation_func = "logistic"

        # 设置默认 alpha
        best_alpha = 0.0001

        # ==========================================================
        # 自动超参数寻优 (Auto-Tuning)
        # ==========================================================
        if self.var_autotune.get():
            self.log("🔧 正在执行自动超参数寻优 (Auto-Tuning)... 请稍候，正在寻找最佳网络结构...")
            try:
                # 1. 定义搜索空间
                param_grid = {
                    'hidden_layer_sizes': [(n_in,), (n_in, n_in), (50,), (100,), (50, 50)],
                    'activation': ['tanh', 'relu', 'logistic'],
                    'alpha': [0.0001, 0.001, 0.01, 0.05]
                }

                # 2. 实例化基础模型
                if mode == "Classification":
                    base_est = MLPClassifier(max_iter=500, random_state=42)
                    scoring_metric = 'accuracy'
                else:
                    base_est = MLPRegressor(max_iter=500, random_state=42)
                    scoring_metric = 'r2'

                # 3. 运行网格搜索 (使用3折交叉验证以节省时间)
                grid = GridSearchCV(base_est, param_grid, cv=3, scoring=scoring_metric, n_jobs=-1)

                if len(X_sc) > 2000:
                    X_tune, _, y_tune, _ = train_test_split(X_sc, y_enc, train_size=0.3, random_state=42)
                    grid.fit(X_tune, y_tune)
                else:
                    grid.fit(X_sc, y_enc)

                # 4. 覆盖原有参数
                best_params = grid.best_params_
                hl_sizes = best_params['hidden_layer_sizes']
                activation_func = best_params['activation']
                best_alpha = best_params['alpha']

                self.log(f"✅ 寻优完成! 最佳参数: 结构={hl_sizes}, 激活={activation_func}, Alpha={best_alpha}")
                self.log("➡️ 系统将使用上述最佳参数进行全流程分析。")

            except Exception as e:
                self.log(f"⚠️ 自动寻优遇到问题 (将回退到Tab1的手动设置): {str(e)}")

        # ==========================================================

        # 4. 训练循环
        table_metrics = []
        table9 = []
        all_preds = []  # 用于收集回归预测值
        best_score = -999
        best_model = None
        final_y_test = None
        final_y_prob = None
        final_y_pred = None
        final_X_test = None

        self.log(
            f"⚙️ 最终模型配置: Layers={hl_sizes}, Act={activation_func}, Alpha={best_alpha}, Train={train_pct * 100:.0f}%")

        for i in range(n_folds):
            X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y_enc, train_size=train_pct, random_state=i * 100)

            # Debug: 查看每一折的数据量
            if self.var_debug.get() and i == 0:
                self.log(f"Fold 1 Split: Train={X_tr.shape}, Test={X_te.shape}", is_debug=True)

            if mode == "Classification":
                model = MLPClassifier(hidden_layer_sizes=hl_sizes, activation=activation_func,
                                      alpha=best_alpha,
                                      solver='adam', max_iter=1000, random_state=i * 42,
                                      early_stopping=self.var_earlystop.get())
                metric_name = "Accuracy"
            else:
                model = MLPRegressor(hidden_layer_sizes=hl_sizes, activation=activation_func,
                                     alpha=best_alpha,
                                     solver='adam', max_iter=1000, random_state=i * 42,
                                     early_stopping=self.var_earlystop.get())
                metric_name = "R2"

            model.fit(X_tr, y_tr)
            pred_te = model.predict(X_te)
            pred_tr = model.predict(X_tr)

            if mode == "Classification":
                # 计算 Accuracy
                score_te = accuracy_score(y_te, pred_te)
                score_tr = accuracy_score(y_tr, pred_tr)

                # 计算 Precision, Recall, F1-Score
                prec, rec, f1, _ = precision_recall_fscore_support(y_te, pred_te, average='weighted', zero_division=0)

                prob_te = model.predict_proba(X_te) if hasattr(model, "predict_proba") else np.zeros((len(y_te), 2))

                table_metrics.append({
                    "Network": f"ANN{i + 1:02d}",
                    "Accuracy_Training": score_tr,
                    "Accuracy_Testing": score_te,
                    "Precision": prec,
                    "Recall": rec,
                    "F1_Score": f1,
                    "Epochs": model.n_iter_
                })

                if score_te > best_score:
                    best_score = score_te;
                    best_model = model
                    final_y_test = y_te;
                    final_y_pred = pred_te;
                    final_y_prob = prob_te;
                    final_X_test = X_te
            else:
                score_te = r2_score(y_te, pred_te)
                rmse_te = np.sqrt(mean_squared_error(y_te, pred_te))
                rmse_tr = np.sqrt(mean_squared_error(y_tr, pred_tr))
                table_metrics.append({"Network": f"ANN{i + 1:02d}", "RMSE_Training": rmse_tr, "RMSE_Testing": rmse_te,
                                      "R2_Testing": score_te, "Epochs": model.n_iter_})

                # [重点修复] 确保收集原始值和预测值
                if scaler_y:
                    y_real = scaler_y.inverse_transform(y_te.reshape(-1, 1)).ravel()
                    p_real = scaler_y.inverse_transform(pred_te.reshape(-1, 1)).ravel()
                    for obs, pre in zip(y_real, p_real):
                        all_preds.append({"Observed": obs, "Predicted": pre, "Fold": i + 1})

                if score_te > best_score:
                    best_score = score_te;
                    best_model = model
                    final_y_test = y_te;
                    final_y_pred = pred_te;
                    final_X_test = X_te

            perm = permutation_importance(model, X_te, y_te, n_repeats=3, random_state=42)
            imp = {"Network": f"ANN{i + 1:02d}"}
            for idx, col in enumerate(inputs): imp[col] = max(0, perm.importances_mean[idx])
            table9.append(imp)
            self.log(f"Fold {i + 1}: {metric_name}={score_te:.4f}")

        # Debug: 最佳模型权重详情
        if self.var_debug.get() and best_model:
            self.log(f"Best Model Iterations: {best_model.n_iter_}", is_debug=True)
            self.log(f"Current Loss: {best_model.loss_:.6f}", is_debug=True)

        # 5. 生成报表
        df_metrics = pd.DataFrame(table_metrics)
        mean_row = df_metrics.drop(columns=['Network']).mean(numeric_only=True)
        sd_row = df_metrics.drop(columns=['Network']).std(numeric_only=True)
        df_metrics.loc['Average'] = mean_row;
        df_metrics.loc['S.D'] = sd_row
        df_metrics.at['Average', 'Network'] = 'Average';
        df_metrics.at['S.D', 'Network'] = 'S.D'
        df_metrics.to_excel(os.path.join(dir_rep, "Table8_Metrics.xlsx"), index=False)

        df_t9 = pd.DataFrame(table9)
        avg_imp = df_t9.drop(columns=['Network']).mean(numeric_only=True)
        max_val = avg_imp.max()
        norm_imp = (avg_imp / max_val * 100) if max_val > 0 else avg_imp * 0
        df_t9.loc['Average'] = avg_imp;
        df_t9.loc['Normalized'] = norm_imp
        df_t9.at['Average', 'Network'] = 'Average Importance';
        df_t9.at['Normalized', 'Network'] = 'Normalized Importance (%)'
        df_t9.to_excel(os.path.join(dir_rep, "Table9_Importance.xlsx"), index=False)

        # 导出最终模型配置参数
        param_records = [
            {"Parameter": "Hidden Layer Sizes", "Value": str(hl_sizes)},
            {"Parameter": "Activation Function", "Value": activation_func},
            {"Parameter": "Alpha (L2 Regularization)", "Value": best_alpha},
            {"Parameter": "Training Split Ratio", "Value": f"{train_pct:.2%}"},
            {"Parameter": "Config Source", "Value": "Auto-Tuning" if self.var_autotune.get() else "Manual Setting"}
        ]
        try:
            pd.DataFrame(param_records).to_excel(os.path.join(dir_rep, "Table_Model_Parameters.xlsx"), index=False)
            self.log("✅ 最终模型参数表已导出: Table_Model_Parameters.xlsx")
        except Exception as e:
            self.log(f"❌ 参数表导出失败: {e}")

        self.log("基础报表生成完毕。")

        # 6. 调用工具类进行可视化
        # A. 敏感性分析
        plt.figure(figsize=(10, 6))
        viz_data = norm_imp.sort_values(ascending=True)
        plt.barh(viz_data.index, viz_data.values, color='teal')
        plt.title("Normalized Importance (Sensitivity Analysis)")
        plt.tight_layout()
        plt.savefig(os.path.join(dir_fig, "Fig_Sensitivity.png"), dpi=300);
        plt.close()

        # B. 神经网络结构 (PNG + HTML)
        try:
            ANN_Visualizer.draw_network_topology(best_model, inputs, target, dir_fig)
            # [Update] 调用新的交互式拓扑图生成方法
            ANN_Visualizer.draw_network_topology_html(best_model, inputs, target, dir_html)
            self.log("✅ 神经网络拓扑图已生成 (PNG + Interactive HTML)")
        except Exception as e:
            self.log(f"拓扑图绘制失败: {e}")

        # C. 模式专属图表
        if mode == "Classification":
            self.log("📊 生成分类专属图表...")
            try:
                ANN_Visualizer.plot_classification_metrics(final_y_test, final_y_prob, final_y_pred, dir_fig, dir_html,
                                                           dir_rep)
            except Exception as e:
                self.log(f"分类图表报错: {e}")
        elif mode == "Regression":
            df_pred = pd.DataFrame(all_preds)
            try:
                if "Fold" in df_pred.columns:
                    df_pred = df_pred[["Fold", "Observed", "Predicted"]]
                df_pred.to_excel(os.path.join(dir_rep, "Table_CV_Predictions_Raw.xlsx"), index=False)
                self.log("✅ 原始预测数据已导出: Table_CV_Predictions_Raw.xlsx")
            except Exception as e:
                self.log(f"❌ 原始数据导出失败: {e}")

            plt.figure(figsize=(8, 8))
            sns.scatterplot(data=df_pred, x='Observed', y='Predicted', alpha=0.5)
            dmin, dmax = df_pred[['Observed', 'Predicted']].min().min(), df_pred[['Observed', 'Predicted']].max().max()
            plt.plot([dmin, dmax], [dmin, dmax], 'r--')
            plt.savefig(os.path.join(dir_fig, "Fig_Obs_vs_Pred.png"), dpi=300);
            plt.close()

        # D. 高级可视化
        if self.var_pdp.get():
            self.log("正在生成 PDP...")
            try:
                fig, ax = plt.subplots(figsize=(12, 8))
                PartialDependenceDisplay.from_estimator(best_model, X_sc, range(min(6, len(inputs))),
                                                        feature_names=inputs, ax=ax)
                plt.savefig(os.path.join(dir_fig, "Fig_PDP.png"), dpi=300);
                plt.close()
                ANN_Visualizer.generate_pdp_html(best_model, X_sc, inputs, dir_html)
            except Exception as e:
                self.log(f"PDP Warning: {e}")

        if self.var_3d.get():
            try:
                top2 = norm_imp.sort_values(ascending=False).head(2).index.tolist()
                ANN_Visualizer.generate_3d_html(best_model, inputs, top2, target, dir_html)
            except:
                pass

        # [MODIFIED] SHAP Analysis Block
        if self.var_shap.get() and SHAP_AVAILABLE and final_X_test is not None:
            self.log("正在计算 SHAP (解释性分析)...")
            try:
                # 1. Calculation
                X_summary = shap.kmeans(X_sc, 10) if len(X_sc) > 100 else X_sc
                explainer = shap.KernelExplainer(best_model.predict, X_summary)

                # Calculate SHAP values
                shap_values_full = explainer.shap_values(final_X_test)

                # Handle Classification vs Regression return types
                if isinstance(shap_values_full, list):
                    # For classification, pick the class of interest (e.g., class 1)
                    sv_to_plot = shap_values_full[1] if len(shap_values_full) > 1 else shap_values_full[0]
                    # Expected value is also a list for classification
                    base_value = explainer.expected_value[1] if len(explainer.expected_value) > 1 else \
                    explainer.expected_value[0]
                else:
                    sv_to_plot = shap_values_full
                    base_value = explainer.expected_value

                # ==========================================================
                # 执行原来的功能 (保持不变)
                # ==========================================================

                # 导出基础 SHAP 值表
                try:
                    df_shap = pd.DataFrame(sv_to_plot, columns=inputs)
                    df_shap.to_excel(os.path.join(dir_rep, "Table_SHAP_Values.xlsx"), index=False)
                except:
                    pass

                # 交互式 HTML 生成
                ANN_Visualizer.generate_shap_html(sv_to_plot, final_X_test, inputs, dir_html, explainer=explainer)
                ANN_Visualizer.draw_shap_correlation_network_html(sv_to_plot, inputs, dir_html, y_values=final_y_test,
                                                                  target_name=target)

                # 依赖图 (您之前添加的功能)
                ANN_Visualizer.draw_shap_dependence_plot(sv_to_plot, final_X_test, inputs, 3, dir_fig)

                # ==========================================================
                # [NEW] 调用新的一站式绘图函数 (满足所有新需求)
                # ==========================================================
                self.log("📊 正在生成高级 SHAP 图表 (Beeswarm, Bar, Waterfall, Force, Decision)...")

                ANN_Visualizer.draw_shap_advanced_plots(
                    shap_values=sv_to_plot,
                    X=final_X_test,
                    feature_names=inputs,
                    base_value=base_value,
                    save_dir=dir_fig
                )

                # 提示：Table 数据通常保存在 Figures 目录或 Reports 目录，这里为了方便查找，
                # draw_shap_advanced_plots 把 Excel 也保存到了 save_dir (即 02_Figures 目录)。
                # 如果您希望数据在 Reports 目录，可以修改上面的 save_dir 为 dir_rep，
                # 但图表需要 save_dir。为了简单，我们将图表和对应数据放在一起，或者您可以手动拆分。

                self.log("✅ 所有 SHAP 高级图表 (600 DPI) 及对应数据已生成")

            except Exception as e:
                self.log(f"SHAP Error: {e}")
                if self.var_debug.get(): traceback.print_exc()

                # 3. Static Plot (PNG)
                plt.figure()
                shap.summary_plot(sv_to_plot, final_X_test, feature_names=inputs, show=False)
                plt.savefig(os.path.join(dir_fig, "Fig_SHAP.png"), dpi=300, bbox_inches='tight');
                plt.close()

                # 4. Interactive HTML (Summary + Force)
                ANN_Visualizer.generate_shap_html(sv_to_plot, final_X_test, inputs, dir_html, explainer=explainer)

                # 5. [New] Interactive Network (Correlation Based)
                # [Fix] Added final_y_test and target name to show target node
                ANN_Visualizer.draw_shap_correlation_network_html(sv_to_plot, inputs, dir_html, y_values=final_y_test,
                                                                  target_name=target)
                self.log("✅ SHAP 交互式图表(Summary + Network) 已生成")

            except Exception as e:
                self.log(f"SHAP Error: {e}")
                if self.var_debug.get(): traceback.print_exc()

        # E. 残差与弹性
        if self.var_resid.get():
            if mode == 'Regression':
                ANN_Visualizer.plot_residuals(final_y_test, final_y_pred, dir_fig, dir_html)
            else:
                self.log("⚠️ 提示: 已跳过残差分析 (仅回归)")

        if self.var_elasticity.get():
            if mode == 'Regression':
                ANN_Analytics.calculate_elasticity(best_model, X_sc, scaler_X, scaler_y, inputs, dir_rep, dir_html,
                                                   self.log)
            else:
                self.log("⚠️ 提示: 已跳过弹性分析 (仅回归)")

        # F. 稳健性检验
        if self.var_robust.get():
            try:
                params = {'hidden_layer_sizes': hl_sizes, 'activation': activation_func, 'solver': 'adam',
                          'max_iter': 1000}
                current_le = le if mode == 'Classification' else None
                ANN_Analytics.run_robustness_check(
                    df_work, inputs, target, scaler_X, scaler_y,
                    params, dir_rep, self.log,
                    mode=mode, le=current_le
                )
            except Exception as e:
                self.log(f"❌ 稳健性检验失败: {e}")
                if self.var_debug.get(): traceback.print_exc()

        # 生成 Word 报告
        try:
            ANN_Reporter.generate_word_report(dir_rep, inputs, target, table_metrics, df_t9, dir_fig,
                                              self.linearity_result)
        except Exception as e:
            self.log(f"报告生成失败: {e}")

        # 保存模型
        if self.var_save_model.get():
            save_pack = {'model': best_model, 'sc_x': scaler_X, 'cols': inputs, 'mode': mode}
            if mode == 'Regression':
                save_pack['sc_y'] = scaler_y
            else:
                save_pack['le'] = le
            joblib.dump(save_pack, os.path.join(dir_mod, "Best_Model.pkl"))

        self.log(f"✅ 分析全部完成! 结果保存在: {res_dir}")
        messagebox.showinfo("Success", f"Done! Check: {res_dir}")
    # ============================================================
    # 预测逻辑 (修复乱码 -> Excel)
    # ============================================================




    def run_prediction(self):
        if not hasattr(self, 'loaded_pack') or not hasattr(self, 'pred_df'):
            messagebox.showwarning("Warn", "请先加载模型和数据");
            return
        try:
            cols = self.loaded_pack['cols']
            X = self.pred_df[cols].values
            X_sc = self.loaded_pack['sc_x'].transform(X)
            y_pred_sc = self.loaded_pack['model'].predict(X_sc)
            y_pred = self.loaded_pack['sc_y'].inverse_transform(y_pred_sc.reshape(-1, 1)).ravel()

            res = self.pred_df.copy()
            res['Predicted_Result'] = y_pred

            save_p = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel Files", "*.xlsx")])
            if save_p:
                res.to_excel(save_p, index=False)
                self.log(f"预测结果已保存至: {save_p}")
                messagebox.showinfo("OK", "预测完成！文件已保存为 Excel 格式。")

        except Exception as e:
            messagebox.showerror("Err", str(e))


if __name__ == "__main__":
    app = SemAnnProApp()
    app.mainloop()
