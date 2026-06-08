import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox, filedialog, colorchooser
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.font_manager as fm
import matplotlib.patheffects as pe
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import math
import platform

# ==========================================
# 0. 高分屏适配
# ==========================================
try:
    import ctypes

    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")
ctk.set_widget_scaling(1.0)
ctk.set_window_scaling(1.0)


# ==========================================
# 1. 数据结构
# ==========================================
class ModelData:
    def __init__(self, name, variables, pls_coeffs, ann_imps):
        self.name = name
        self.variables = variables
        self.pls_coeffs = pls_coeffs
        self.ann_imps = ann_imps


# ==========================================
# 2. 弹窗：添加/编辑模型
# ==========================================
class ModelDialog(ctk.CTkToplevel):
    def __init__(self, parent, model_data=None):
        super().__init__(parent)
        self.title("编辑模型")
        self.geometry("500x550")
        self.result = None
        self.transient(parent)
        self.grab_set()

        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(frame, text="模型配置", font=("Arial", 20, "bold")).pack(anchor="w", pady=(0, 10))

        ctk.CTkLabel(frame, text="模型名称:", font=("Arial", 14)).pack(anchor="w")
        self.name_entry = ctk.CTkEntry(frame, placeholder_text="例如: Model A")
        if model_data: self.name_entry.insert(0, model_data.name)
        self.name_entry.pack(fill="x", pady=(5, 15))

        ctk.CTkLabel(frame, text="数据 (格式: 变量, PLS, ANN):", font=("Arial", 14)).pack(anchor="w")
        ctk.CTkLabel(frame, text="例如: IO, 0.312, 100.0", text_color="gray", font=("Arial", 12)).pack(anchor="w",
                                                                                                       pady=(0, 5))

        self.text_area = ctk.CTkTextbox(frame, height=250)
        self.text_area.pack(fill="both", expand=True, pady=5)

        if model_data:
            lines = [f"{v}, {p}, {a}" for v, p, a in
                     zip(model_data.variables, model_data.pls_coeffs, model_data.ann_imps)]
            self.text_area.insert("0.0", "\n".join(lines))

        btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
        btn_frame.pack(fill="x", pady=10)

        ctk.CTkButton(btn_frame, text="保存", command=self.save_data, width=100).pack(side="right", padx=5)
        ctk.CTkButton(btn_frame, text="取消", command=self.destroy, fg_color="transparent", border_width=1,
                      text_color="gray", width=80).pack(side="right")

    def save_data(self):
        name = self.name_entry.get().strip()
        raw_text = self.text_area.get("0.0", "end").strip()
        if not name:
            tk.messagebox.showerror("错误", "模型名称不能为空")
            return
        vars_list, pls_list, ann_list = [], [], []
        try:
            for line in raw_text.split('\n'):
                line = line.strip()
                if not line: continue
                parts = line.split(',')
                if len(parts) != 3: raise ValueError(f"格式错误行: {line}")
                vars_list.append(parts[0].strip())
                pls_list.append(float(parts[1].strip()))
                ann_list.append(float(parts[2].strip()))
            self.result = ModelData(name, vars_list, pls_list, ann_list)
            self.destroy()
        except Exception as e:
            tk.messagebox.showerror("解析错误", f"请检查数据格式。\n{str(e)}")


# ==========================================
# 3. 主程序 GUI
# ==========================================
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("ANN & PLS Visualizer V6.4 (Layer Fix)")
        self.geometry("1400x950")

        self.models = self.get_initial_data()

        self.font_alias_map = {
            "宋体": "SimSun", "黑体": "SimHei", "微软雅黑": "Microsoft YaHei",
            "楷体": "KaiTi", "仿宋": "FangSong", "隶书": "LiSu", "幼圆": "YouYuan",
            "Times New Roman": "Times New Roman", "Arial": "Arial"
        }
        self.system_fonts = sorted(list(set([f.name for f in fm.fontManager.ttflist])))
        self.display_fonts = list(self.font_alias_map.keys()) + self.system_fonts

        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # 1. Sidebar
        self.sidebar_container = ctk.CTkFrame(self, width=420, corner_radius=0)
        self.sidebar_container.grid(row=0, column=0, sticky="nsew")
        self.sidebar_container.pack_propagate(False)

        ctk.CTkLabel(self.sidebar_container, text="配置面板", font=("Arial", 24, "bold")).pack(pady=(20, 10), padx=20,
                                                                                               anchor="w")

        self.scroll_frame = ctk.CTkScrollableFrame(self.sidebar_container, fg_color="transparent")
        self.scroll_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.setup_controls()

        # 2. Canvas Area
        self.right_frame = ctk.CTkFrame(self, fg_color="white")
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        self.canvas_container = ctk.CTkFrame(self.right_frame, fg_color="white")
        self.canvas_container.pack(side="top", fill="both", expand=True)

        self.toolbar_container = ctk.CTkFrame(self.right_frame, height=40, fg_color="#F0F0F0")
        self.toolbar_container.pack(side="bottom", fill="x")

        self.fig = None
        self.plot_figure()

    def get_initial_data(self):
        data = []
        raw = {
            'Model A': {'vars': ['IO', 'SO', 'CO', 'SFO'], 'pls': [0.312, 0.284, 0.207, 0.183],
                        'ann': [100.00, 71.33, 42.51, 45.45]},
            'Model B': {'vars': ['ICA', 'ICO'], 'pls': [0.469, 0.305], 'ann': [100.00, 39.51]},
            'Model C': {'vars': ['HPR', 'IBI', 'SID'], 'pls': [0.643, 0.517, 0.443], 'ann': [100.00, 61.68, 48.52]},
            'Model D': {'vars': ['IIG', 'VUL', 'BCN'], 'pls': [0.408, 0.408, 0.406], 'ann': [100.00, 62.74, 86.27]}
        }
        for name, content in raw.items():
            data.append(ModelData(name, content['vars'], content['pls'], content['ann']))
        return data

    def setup_controls(self):
        self.add_section_label("视图模式")
        self.view_mode = ctk.CTkSegmentedButton(self.scroll_frame, values=["双轴 (Both)", "仅 ANN", "仅 PLS"],
                                                command=self.on_mode_change)
        self.view_mode.set("双轴 (Both)")
        self.view_mode.pack(padx=10, pady=5, fill="x")

        self.add_section_label("模型管理")
        self.model_list_frame = ctk.CTkFrame(self.scroll_frame, fg_color=("gray90", "gray30"))
        self.model_list_frame.pack(padx=10, pady=5, fill="x")
        self.refresh_model_list()

        ctk.CTkButton(self.scroll_frame, text="+ 添加新模型", command=self.add_model).pack(padx=10, pady=10, fill="x")
        ctk.CTkProgressBar(self.scroll_frame, height=2).pack(padx=10, pady=15, fill="x")

        self.add_section_label("颜色设置")
        self.color_ann_var = tk.StringVar(value="#008080")
        self.color_pls_var = tk.StringVar(value="#D35400")
        self.color_ann_text_var = tk.StringVar(value="#333333")
        self.color_pls_text_var = tk.StringVar(value="#333333")
        self.color_axis_text_var = tk.StringVar(value="#000000")

        self.create_color_picker("ANN 颜色:", self.color_ann_var)
        self.create_color_picker("PLS 颜色:", self.color_pls_var)
        self.create_color_picker("ANN 数值颜色:", self.color_ann_text_var)
        self.create_color_picker("PLS 数值颜色:", self.color_pls_text_var)
        self.create_color_picker("坐标文字颜色:", self.color_axis_text_var)

        self.add_section_label("字体与尺寸")
        frame_font = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
        frame_font.pack(padx=10, pady=2, fill="x")
        ctk.CTkLabel(frame_font, text="全局字体:", width=80, anchor="w").pack(side="left")
        self.font_var = tk.StringVar(value="Times New Roman")
        self.font_combo = ctk.CTkComboBox(frame_font, values=self.display_fonts, variable=self.font_var)
        self.font_combo.pack(side="right", expand=True, fill="x")
        self.font_combo._entry.bind("<KeyRelease>", self.filter_fonts)

        self.val_label_fs = self.create_safe_entry("数值标签字号:", "12")
        self.axis_fs = self.create_safe_entry("坐标轴字号:", "12")
        self.model_title_fs = self.create_safe_entry("模型名称字号:", "14")
        self.legend_fs = self.create_safe_entry("图例文字大小:", "12")
        self.legend_marker_size = self.create_safe_entry("图例图标大小:", "8")

        self.add_section_label("导出设置")
        self.dpi_val = self.create_safe_entry("导出 DPI:", "600")

        ctk.CTkLabel(self.scroll_frame, text="").pack(pady=10)
        ctk.CTkButton(self.scroll_frame, text="⟳ 刷新预览", command=lambda: self.plot_figure(save_mode=False),
                      fg_color="#2ECC71", hover_color="#27AE60").pack(padx=10, pady=5, fill="x")
        ctk.CTkButton(self.scroll_frame, text="💾 导出高清图", command=self.save_figure, fg_color="#3498DB",
                      hover_color="#2980B9").pack(padx=10, pady=5, fill="x")
        ctk.CTkButton(self.scroll_frame, text="⚠ 重置所有参数", command=self.reset_params, fg_color="#E74C3C",
                      hover_color="#C0392B").pack(padx=10, pady=(15, 30), fill="x")

    def create_safe_entry(self, label, default_val):
        frame = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
        frame.pack(padx=10, pady=2, fill="x")
        ctk.CTkLabel(frame, text=label, width=110, anchor="w").pack(side="left")
        var = tk.StringVar(value=default_val)
        ctk.CTkEntry(frame, textvariable=var).pack(side="right", expand=True, fill="x")
        return var

    def get_safe_int(self, var, default):
        try:
            val = var.get().strip()
            if not val: return default
            return int(val)
        except ValueError:
            return default

    def filter_fonts(self, event):
        text = self.font_var.get().lower()
        if text == "":
            self.font_combo.configure(values=self.display_fonts)
        else:
            filtered = [f for f in self.display_fonts if text in f.lower()]
            self.font_combo.configure(values=filtered)

    def add_section_label(self, text):
        ctk.CTkLabel(self.scroll_frame, text=text, font=("Arial", 14, "bold")).pack(pady=(15, 5), padx=10, anchor="w")

    def create_color_picker(self, label, var):
        frame = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
        frame.pack(padx=10, pady=2, fill="x")
        ctk.CTkLabel(frame, text=label, width=110, anchor="w").pack(side="left")
        btn = ctk.CTkButton(frame, text=var.get(), fg_color=var.get(), width=80,
                            command=lambda: self.pick_color(var, btn))
        btn.pack(side="right", expand=True, fill="x")

    def pick_color(self, var, btn):
        color_code = colorchooser.askcolor(color=var.get(), title="选择颜色")[1]
        if color_code:
            var.set(color_code)
            btn.configure(fg_color=color_code, text=color_code)
            self.plot_figure(save_mode=False)

    def reset_params(self):
        if messagebox.askyesno("重置", "确定要恢复默认颜色和字号设置吗？"):
            self.color_ann_var.set("#008080")
            self.color_pls_var.set("#D35400")
            self.color_ann_text_var.set("#333333")
            self.color_pls_text_var.set("#333333")
            self.color_axis_text_var.set("#000000")
            self.font_var.set("Times New Roman")
            self.val_label_fs.set("12")
            self.axis_fs.set("12")
            self.model_title_fs.set("14")
            self.legend_fs.set("12")
            self.legend_marker_size.set("8")
            self.dpi_val.set("600")
            self.plot_figure()
            messagebox.showinfo("提示", "参数已重置。")

    def refresh_model_list(self):
        for widget in self.model_list_frame.winfo_children():
            widget.destroy()
        for i, model in enumerate(self.models):
            card = ctk.CTkFrame(self.model_list_frame, fg_color="transparent")
            card.pack(fill="x", pady=2)
            bg = ctk.CTkFrame(card, fg_color=("white", "gray40"))
            bg.pack(fill="x", padx=0, pady=0)
            ctk.CTkLabel(bg, text=f"{i + 1}. {model.name}", font=("Arial", 12)).pack(side="left", padx=10, pady=8)
            ctk.CTkButton(bg, text="×", width=30, height=25, fg_color="#E74C3C", hover_color="#C0392B",
                          command=lambda idx=i: self.delete_model(idx)).pack(side="right", padx=5)
            ctk.CTkButton(bg, text="✎", width=30, height=25, fg_color="#F39C12", hover_color="#D35400",
                          command=lambda idx=i: self.edit_model(idx)).pack(side="right", padx=0)

    def on_mode_change(self, value):
        self.plot_figure()

    def add_model(self):
        dialog = ModelDialog(self)
        self.wait_window(dialog)
        if dialog.result:
            self.models.append(dialog.result)
            self.refresh_model_list()
            self.plot_figure()

    def edit_model(self, idx):
        dialog = ModelDialog(self, self.models[idx])
        self.wait_window(dialog)
        if dialog.result:
            self.models[idx] = dialog.result
            self.refresh_model_list()
            self.plot_figure()

    def delete_model(self, idx):
        if messagebox.askyesno("确认", "确定删除该模型吗？"):
            del self.models[idx]
            self.refresh_model_list()
            self.plot_figure()

    def plot_figure(self, save_mode=False):
        if self.fig:
            plt.close(self.fig)

        for widget in self.canvas_container.winfo_children(): widget.destroy()
        for widget in self.toolbar_container.winfo_children(): widget.destroy()

        c_ann = self.color_ann_var.get()
        c_pls = self.color_pls_var.get()
        c_ann_text = self.color_ann_text_var.get()
        c_pls_text = self.color_pls_text_var.get()
        c_axis_text = self.color_axis_text_var.get()

        raw_font = self.font_var.get()
        actual_font = self.font_alias_map.get(raw_font, raw_font)

        val_fs = self.get_safe_int(self.val_label_fs, 12)
        ax_fs = self.get_safe_int(self.axis_fs, 12)
        title_fs = self.get_safe_int(self.model_title_fs, 14)
        leg_fs = self.get_safe_int(self.legend_fs, 12)
        leg_mk_size = self.get_safe_int(self.legend_marker_size, 8)

        current_mode = self.view_mode.get()
        n = len(self.models)
        if n == 0: return
        cols = 2 if n > 1 else 1
        rows = math.ceil(n / cols)
        figsize = (12, 5 * rows)

        sns.set_theme(style="white")
        try:
            plt.rcParams['font.family'] = actual_font
            plt.rcParams['axes.unicode_minus'] = False
        except:
            pass

        plt.rcParams['font.size'] = ax_fs

        self.fig, axes = plt.subplots(rows, cols, figsize=figsize)
        dynamic_hspace = 0.6 if rows > 2 else 0.4
        plt.subplots_adjust(wspace=0.3, hspace=dynamic_hspace, top=0.88 if len(self.models) > 1 else 0.85, bottom=0.1)

        axes_flat = [axes] if n == 1 else axes.flatten()

        # 定义描边
        text_halo = [pe.withStroke(linewidth=3, foreground="white")]

        for i, model in enumerate(self.models):
            ax1 = axes_flat[i]
            df = pd.DataFrame({'Variable': model.variables, 'PLS': model.pls_coeffs, 'ANN': model.ann_imps})
            df = df.sort_values(by='ANN', ascending=True)
            y_pos = np.arange(len(df))

            # ---------------------
            # 1. 绘制 ANN (柱状图)
            # ---------------------
            bars = None
            if current_mode in ["双轴 (Both)", "仅 ANN"]:
                bars = ax1.barh(y_pos, df['ANN'], height=0.5, color=c_ann, alpha=0.7)
                ax1.set_xlim(0, 115)
                ax1.tick_params(axis='x', labelsize=ax_fs, colors=c_axis_text)

            # ---------------------
            # 2. 绘制 PLS (折线图)
            # ---------------------
            ax2 = None
            if current_mode == "双轴 (Both)":
                ax2 = ax1.twiny()
                ax2.plot(df['PLS'], y_pos, color=c_pls, marker='D', markersize=8, linewidth=3, linestyle='--')
                max_p = df['PLS'].max()
                ax2.set_xlim(0, max_p * 1.5 if max_p > 0 else 1.0)
                ax2.tick_params(axis='x', labelsize=ax_fs, colors=c_axis_text)

                # 去除边框
                for spine in ax2.spines.values(): spine.set_visible(False)

                # --- 关键修改：PLS 标签绘制在 ax2 上 ---
                for j, val in enumerate(df['PLS']):
                    t = ax2.text(val + 0.01, j + 0.25, f'{val:.3f}', va='center',
                                 fontsize=val_fs, color=c_pls_text, fontweight='bold', zorder=20)
                    t.set_path_effects(text_halo)

            elif current_mode == "仅 PLS":
                ax1.plot(df['PLS'], y_pos, color=c_pls, marker='D', markersize=8, linewidth=3, linestyle='--')
                for j, val in enumerate(df['PLS']):
                    t = ax1.text(val + 0.01, j, f'{val:.3f}', va='bottom',
                                 fontsize=val_fs, color=c_pls_text, fontweight='bold', zorder=20)
                    t.set_path_effects(text_halo)
                max_p = df['PLS'].max()
                ax1.set_xlim(0, max_p * 1.3 if max_p > 0 else 1.0)
                ax1.tick_params(axis='x', labelsize=ax_fs, colors=c_axis_text)

            # ---------------------
            # 3. 绘制 ANN 标签 (防遮挡核心逻辑)
            # ---------------------
            if bars:
                for bar in bars:
                    # 如果有 ax2 (双轴模式)，我们将 ANN 标签也画在 ax2 上
                    # 这样它就会处于顶层，盖住折线。
                    # 关键点：使用 transform=ax1.transData 保持位置正确
                    target_ax = ax2 if (ax2 is not None) else ax1

                    # 坐标
                    x_pos = bar.get_width() + 2
                    y_coord = bar.get_y() + bar.get_height() / 2

                    # 绘制
                    t = target_ax.text(x_pos, y_coord, f'{bar.get_width():.1f}%', va='center',
                                       fontsize=val_fs, color=c_ann_text, fontweight='bold', zorder=30,
                                       transform=ax1.transData)  # 强制使用 ax1 的坐标系

                    t.set_path_effects(text_halo)

            # 公共轴设置
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(df['Variable'], fontsize=ax_fs, fontweight='bold', color=c_axis_text)
            ax1.set_title("")
            ax1.set_xlabel(model.name, fontsize=title_fs, fontweight='bold', labelpad=15, color='#333333')

            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            if current_mode == "仅 ANN":
                ax1.spines['bottom'].set_visible(True)
            elif current_mode == "双轴 (Both)":
                ax1.spines['bottom'].set_visible(False)

        for j in range(n, len(axes_flat)): axes_flat[j].axis('off')

        handles = []
        if current_mode in ["双轴 (Both)", "仅 ANN"]:
            handles.append(mpatches.Patch(color=c_ann, label='ANN Normalized Importance (%)'))
        if current_mode in ["双轴 (Both)", "仅 PLS"]:
            handles.append(mlines.Line2D([], [], color=c_pls, marker='D', markersize=leg_mk_size, linestyle='--',
                                         label='PLS-SEM Path Coefficient (β)'))

        self.fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2, fontsize=leg_fs,
                        frameon=False)

        if not save_mode:
            canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_container)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            toolbar = NavigationToolbar2Tk(canvas, self.toolbar_container)
            toolbar.update()
            toolbar.pack(side="bottom", fill="x")

    def save_figure(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG Image", "*.png"), ("PDF", "*.pdf")])
        if file_path:
            try:
                self.plot_figure(save_mode=True)
                dpi_val = self.get_safe_int(self.dpi_val, 600)
                self.fig.savefig(file_path, dpi=dpi_val, bbox_inches='tight')
                messagebox.showinfo("成功", f"图片已保存")
                self.plot_figure(save_mode=False)
            except Exception as e:
                messagebox.showerror("错误", str(e))


if __name__ == "__main__":
    app = App()
    app.mainloop()