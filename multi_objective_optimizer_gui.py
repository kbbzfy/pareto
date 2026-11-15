import sys
import os
import subprocess
import json
import yaml
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout,
                            QHBoxLayout, QLabel, QLineEdit, QComboBox, QPushButton,
                            QSpinBox, QTableWidget, QTableWidgetItem, QFileDialog,
                            QGroupBox, QFormLayout, QCheckBox, QRadioButton, QScrollArea,
                            QSplitter, QTextEdit, QStatusBar, QMessageBox, QListWidget, QDoubleSpinBox, QDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QIcon, QPixmap, QTextCursor
from datetime import datetime
from src.problems.tsp import TSP_OBJECTIVE_NAMES

class MultiObjectiveOptimizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("多目标优化框架")
        self.setGeometry(100, 100, 1200, 800)
        
        # 设置日志文件
        self.setup_file_logging()
        
        # 创建中心部件和布局
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        
        # 创建左侧控制面板
        self.create_control_panel()
        
        # 创建右侧内容区
        self.create_content_area()
        
        # 创建状态栏
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("就绪")
        
        # 日志缓冲区和计时器
        self.log_buffer = []
        self.log_timer = QTimer(self)
        self.log_timer.timeout.connect(self.flush_log_buffer)
        self.log_timer.start(200)  # 每200毫秒更新一次UI
        
        # 初始化配置
        self.load_default_config()
        
        # 设置初始问题类型UI
        self.problem_type_changed(self.problem_type.currentText())
    
    def setup_file_logging(self):
        """设置文件日志"""
        import logging
        # 确保logs目录存在
        os.makedirs("gui_logs", exist_ok=True)
        
        # 创建具有时间戳的日志文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join("gui_logs", f"gui_{timestamp}.log")
        
        # 配置日志记录器
        self.file_logger = logging.getLogger("gui_logger")
        self.file_logger.setLevel(logging.INFO)
        
        # 创建文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # 添加处理器到记录器
        self.file_logger.addHandler(file_handler)
        self.file_logger.info("GUI会话已启动")
    
    def create_control_panel(self):
        # 创建左侧控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        
        # 1. 问题类型选择
        problem_group = QGroupBox("问题类型")
        problem_layout = QFormLayout()
        
        # 创建问题类型下拉框
        self.problem_type = QComboBox()
        self.problem_type.addItems(["knapsack", "tsp"])
        self.problem_type.currentTextChanged.connect(self.problem_type_changed)
        
        # 创建类别数量和每类物品数控件
        self.num_categories = QSpinBox()
        self.num_categories.setRange(1, 20)
        self.num_categories.setValue(5)
        
        self.items_per_category = QSpinBox()
        self.items_per_category.setRange(1, 50)
        self.items_per_category.setValue(10)
        
        # 创建标签
        self.num_categories_label = QLabel("类别数量:")
        self.items_per_category_label = QLabel("每类物品数:")
        
        # 添加控件到布局
        problem_layout.addRow("问题类型:", self.problem_type)
        problem_layout.addRow(self.num_categories_label, self.num_categories)
        problem_layout.addRow(self.items_per_category_label, self.items_per_category)
        
        problem_group.setLayout(problem_layout)
        control_layout.addWidget(problem_group)
        
        # 2. 运行模式选择
        mode_group = QGroupBox("运行模式")
        mode_layout = QVBoxLayout()
        
        # 生成数据按钮
        generate_button = QPushButton("生成数据")
        generate_button.clicked.connect(lambda: self.run_optimization("generate"))
        mode_layout.addWidget(generate_button)
        
        # 训练模型按钮
        train_button = QPushButton("训练模型")
        train_button.clicked.connect(lambda: self.run_optimization("train"))
        mode_layout.addWidget(train_button)
        
        # 测试模型按钮
        test_button = QPushButton("测试模型")
        test_button.clicked.connect(lambda: self.run_optimization("test"))
        mode_layout.addWidget(test_button)
        
        # 取消按钮
        cancel_button = QPushButton("取消操作")
        cancel_button.setEnabled(False)
        cancel_button.clicked.connect(self.cancel_operation)
        self.cancel_button = cancel_button  # 保存引用以便稍后启用/禁用
        mode_layout.addWidget(cancel_button)
        
        mode_group.setLayout(mode_layout)
        control_layout.addWidget(mode_group)
        
        # 3. 目标设置
        objectives_group = QGroupBox("目标设置")
        objectives_layout = QVBoxLayout()
        
        obj_count_layout = QHBoxLayout()
        obj_count_layout.addWidget(QLabel("目标数量:"))
        self.n_objectives = QSpinBox()
        self.n_objectives.setRange(4, 10)
        self.n_objectives.setValue(4)
        self.n_objectives.valueChanged.connect(self.update_objectives_table)
        obj_count_layout.addWidget(self.n_objectives)
        objectives_layout.addLayout(obj_count_layout)
        
        # 目标表格
        self.objectives_table = QTableWidget(4, 6)  # 增加约束值列
        self.objectives_table.setHorizontalHeaderLabels(["目标名称", "类型", "约束", "约束值", "最小值", "最大值"])
        self.objectives_table.setColumnWidth(0, 100)  # 名称列宽
        self.objectives_table.setColumnWidth(1, 80)   # 类型列宽
        self.objectives_table.setColumnWidth(2, 80)   # 约束列宽
        self.objectives_table.setColumnWidth(3, 80)   # 约束值列宽
        self.objectives_table.setColumnWidth(4, 80)   # 最小值列宽
        self.objectives_table.setColumnWidth(5, 80)   # 最大值列宽
        self.update_objectives_table()
        objectives_layout.addWidget(self.objectives_table)
        
        objectives_group.setLayout(objectives_layout)
        control_layout.addWidget(objectives_group)
        
        # 4. 模型参数
        model_group = QGroupBox("模型参数")
        model_layout = QFormLayout()
        self.hidden_dim = QSpinBox()
        self.hidden_dim.setRange(32, 1024)
        self.hidden_dim.setValue(256)
        self.hidden_dim.setSingleStep(32)
        
        self.entropy_weight = QDoubleSpinBox()
        self.entropy_weight.setRange(0.001, 0.1)
        self.entropy_weight.setValue(0.01)
        self.entropy_weight.setSingleStep(0.001)
        self.entropy_weight.setDecimals(3)
        
        self.value_weight = QDoubleSpinBox()
        self.value_weight.setRange(0.1, 1.0)
        self.value_weight.setValue(0.5)
        self.value_weight.setSingleStep(0.1)
        self.value_weight.setDecimals(2)
        
        model_layout.addRow("隐藏层维度:", self.hidden_dim)
        model_layout.addRow("策略熵权重:", self.entropy_weight)
        model_layout.addRow("价值函数权重:", self.value_weight)
        model_group.setLayout(model_layout)
        control_layout.addWidget(model_group)
        
        # 5. 检查点选择
        checkpoint_group = QGroupBox("检查点")
        checkpoint_layout = QHBoxLayout()
        self.checkpoint_path = QLineEdit()
        self.checkpoint_path.setReadOnly(True)
        self.browse_button = QPushButton("浏览...")
        self.browse_button.clicked.connect(self.browse_checkpoint)
        checkpoint_layout.addWidget(self.checkpoint_path)
        checkpoint_layout.addWidget(self.browse_button)
        checkpoint_group.setLayout(checkpoint_layout)
        control_layout.addWidget(checkpoint_group)
        
        # 在create_control_panel方法中添加检查点目录设置
        checkpoint_dir_layout = QHBoxLayout()
        checkpoint_dir_layout.addWidget(QLabel("检查点保存目录:"))
        self.checkpoint_dir = QLineEdit("checkpoints")
        browse_checkpoint_dir_button = QPushButton("浏览...")
        browse_checkpoint_dir_button.clicked.connect(self.browse_checkpoint_dir)
        checkpoint_dir_layout.addWidget(self.checkpoint_dir)
        checkpoint_dir_layout.addWidget(browse_checkpoint_dir_button)
        checkpoint_group.layout().addLayout(checkpoint_dir_layout)
        
        # 添加伸缩控件使控制面板保持在顶部
        control_layout.addStretch()
        
        # 将控制面板添加到主布局
        self.main_layout.addWidget(control_panel, 1)
    
    def create_content_area(self):
        # 创建右侧内容区
        content_area = QTabWidget()
        
        # 1. 日志选项卡
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        log_layout.addWidget(QLabel("运行日志:"))
        log_layout.addWidget(self.log_output)
        
        log_buttons = QHBoxLayout()
        clear_log_button = QPushButton("清除日志")
        clear_log_button.clicked.connect(self.clear_log)
        log_buttons.addWidget(clear_log_button)
        log_layout.addLayout(log_buttons)
        
        content_area.addTab(log_tab, "日志")
        
        # 2. 可视化选项卡
        viz_tab = QWidget()
        viz_layout = QVBoxLayout(viz_tab)
        
        # 创建可视化图像容器（使用滚动区域，方便查看多个图表）
        self.viz_scroll_area = QScrollArea()
        self.viz_scroll_area.setWidgetResizable(True)
        
        # 创建一个容器widget来包含所有图像
        self.viz_container = QWidget()
        self.viz_container_layout = QVBoxLayout(self.viz_container)
        
        # 为每种图表创建一个标签
        self.viz_images = {}
        viz_types = ["pareto", "parallel", "radar", "heatmap", "tsp_route", "tsp_front"]
        
        for viz_type in viz_types:
            viz_group = QGroupBox(self.get_viz_title(viz_type))
            viz_group_layout = QVBoxLayout(viz_group)
            
            img_label = QLabel()
            img_label.setAlignment(Qt.AlignCenter)
            img_label.setMinimumSize(500, 400)
            img_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ddd;")
            
            viz_group_layout.addWidget(img_label)
            self.viz_images[viz_type] = img_label
            
            self.viz_container_layout.addWidget(viz_group)
        
        self.viz_scroll_area.setWidget(self.viz_container)
        viz_layout.addWidget(self.viz_scroll_area)
        
        # 刷新按钮
        viz_buttons = QHBoxLayout()
        refresh_viz_button = QPushButton("刷新可视化")
        refresh_viz_button.clicked.connect(self.refresh_visualization)
        viz_buttons.addWidget(refresh_viz_button)
        viz_layout.addLayout(viz_buttons)
        
        content_area.addTab(viz_tab, "可视化")
        
        # 3. 数据查看选项卡
        data_tab = QWidget()
        data_layout = QVBoxLayout(data_tab)
        
        # 添加标题和说明
        title_label = QLabel("生成的数据")
        title_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        data_layout.addWidget(title_label)
        
        # 创建数据表格
        self.data_table = QTableWidget()
        
        # 默认设置为背包问题的表头
        # 注意：这里的实际初始化会在problem_type_changed中完成
        self.data_table.setColumnCount(7)  # ID, 类别, 价值, 重量, 体积, 成本, 是否选中
        self.data_table.setHorizontalHeaderLabels(['ID', '类别', '价值', '重量', '体积', '成本', '是否选中'])
        
        # 设置表格样式
        self.data_table.setAlternatingRowColors(True)  # 交替行颜色
        self.data_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #d0d0d0;
                selection-background-color: #0078d7;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                padding: 5px;
                border: 1px solid #d0d0d0;
                font-weight: bold;
            }
        """)
        
        # 创建一个滚动区域来容纳表格
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.data_table)
        scroll_area.setWidgetResizable(True)
        data_layout.addWidget(scroll_area)
        
        # 添加控制按钮
        button_layout = QHBoxLayout()
        load_data_button = QPushButton("刷新数据")
        load_data_button.clicked.connect(self.load_generated_data)
        button_layout.addWidget(load_data_button)
        
        export_button = QPushButton("导出数据")
        export_button.clicked.connect(self.export_data)
        button_layout.addWidget(export_button)
        
        # 添加问题类型指示标签
        self.data_type_label = QLabel("当前数据类型: 背包问题")
        self.data_type_label.setStyleSheet("color: #666; font-style: italic;")
        button_layout.addWidget(self.data_type_label)
        
        button_layout.addStretch()
        data_layout.addLayout(button_layout)
        
        content_area.addTab(data_tab, "数据查看")
        
        # 将内容区添加到主布局，修改占比为1:1
        self.main_layout.addWidget(content_area, 1)
    
    def update_objectives_table(self):
        """更新目标表格"""
        n = self.n_objectives.value()
        self.objectives_table.setRowCount(n)
        
        # 设置表格列
        self.objectives_table.setColumnCount(6)
        self.objectives_table.setHorizontalHeaderLabels(["目标名称", "类型", "约束", "约束值", "最小值", "最大值"])
        
        # 默认值
        is_tsp = self.problem_type.currentText() == "tsp"
        default_names = TSP_OBJECTIVE_NAMES if is_tsp else ["价值", "重量", "体积", "成本"]
        default_min_values = ["20", "5", "10", "8"]
        default_max_values = ["300", "60", "150", "90"]
        
        for i in range(n):
            # 目标名称
            name = default_names[i] if i < len(default_names) else f"目标{i+1}"
            name_item = QTableWidgetItem(name)
            self.objectives_table.setItem(i, 0, name_item)
            
            # 目标类型
            type_combo = QComboBox()
            type_combo.addItems(["最大化", "最小化"])
            type_combo.setCurrentText("最小化" if is_tsp or i > 0 else "最大化")
            self.objectives_table.setCellWidget(i, 1, type_combo)
            
            # 约束设置
            constraint_combo = QComboBox()
            constraint_combo.addItems(["是", "否"])
            constraint_combo.setCurrentText("否" if is_tsp or i == 0 else "是")
            constraint_combo.setEnabled(not is_tsp)
            constraint_combo.currentTextChanged.connect(lambda text, row=i: self.on_constraint_changed(text, row))
            self.objectives_table.setCellWidget(i, 2, constraint_combo)
            
            # 约束值
            constraint_value = QLineEdit()
            constraint_value.setText("100" if i > 0 and not is_tsp else "N/A")
            constraint_value.setEnabled(i > 0 and not is_tsp)
            self.objectives_table.setCellWidget(i, 3, constraint_value)
            
            # 最小值和最大值
            if i < len(default_min_values):
                min_value = QLineEdit(default_min_values[i])
                max_value = QLineEdit(default_max_values[i])
            else:
                min_value = QLineEdit("0")
                max_value = QLineEdit("100")
            
            self.objectives_table.setCellWidget(i, 4, min_value)
            self.objectives_table.setCellWidget(i, 5, max_value)
    
    def on_constraint_changed(self, text, row):
        """处理约束状态改变"""
        constraint_value = self.objectives_table.cellWidget(row, 3)
        if constraint_value:
            if text == "是":
                constraint_value.setEnabled(True)
                if constraint_value.text() == "N/A":
                    # 设置默认约束值
                    if row == 0:  # 价值
                        constraint_value.setText("1000")
                    else:  # 重量、体积、成本等
                        constraint_value.setText("100")
            else:
                constraint_value.setEnabled(False)
                constraint_value.setText("N/A")
    
    def browse_checkpoint(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择检查点文件", "checkpoints", "PyTorch模型 (*.pt)")
        if file_path:
            self.checkpoint_path.setText(file_path)
    
    def browse_checkpoint_dir(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择检查点保存目录", ".")
        if dir_path:
            self.checkpoint_dir.setText(dir_path)
    
    def load_default_config(self):
        """初始化基本配置"""
        # 不再使用config_editor，因为配置标签页已删除
        pass
    
    def load_config(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "加载配置文件", "", "YAML文件 (*.yaml *.yml)")
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    config_text = f.read()
                    config = yaml.safe_load(config_text)
                    self.config_editor.setText(config_text)
                    
                    # 更新UI控件
                    if 'problem' in config and 'type' in config['problem']:
                        index = self.problem_type.findText(config['problem']['type'])
                        if index >= 0:
                            self.problem_type.setCurrentIndex(index)
                    
                    if 'model' in config:
                        if 'n_objectives' in config['model']:
                            self.n_objectives.setValue(config['model']['n_objectives'])
                        
                        if 'hidden_dim' in config['model']:
                            self.hidden_dim.setValue(config['model']['hidden_dim'])
                        
                        if 'objective_names' in config['model'] and 'objective_types' in config['model']:
                            names = config['model']['objective_names']
                            types = config['model']['objective_types']
                            n = min(len(names), len(types), self.objectives_table.rowCount())
                            
                            for i in range(n):
                                self.objectives_table.setItem(i, 0, QTableWidgetItem(names[i]))
                                type_combo = self.objectives_table.cellWidget(i, 1)
                                type_combo.setCurrentIndex(0 if types[i] == 1 else 1)
                
                self.statusBar.showMessage(f"已加载配置: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "加载错误", f"加载配置文件时出错: {str(e)}")
    
    def clear_log(self):
        self.log_output.clear()
    
    def get_viz_title(self, viz_type):
        """获取可视化类型的标题"""
        titles = {
            "pareto": "散点矩阵图",
            "parallel": "平行坐标图",
            "radar": "雷达图",
            "heatmap": "热力图",
            "tsp_route": "TSP 闭合路线",
            "tsp_front": "TSP 候选帕累托前沿"
        }
        return titles.get(viz_type, viz_type.title())
    
    def change_visualization(self, index):
        """切换显示的可视化图表"""
        selected = self.viz_selector.currentText()
        
        # 控制各图表的可见性
        if selected == "所有图表":
            # 显示所有图表
            for viz_type, img_label in self.viz_images.items():
                img_label.parent().setVisible(True)
        else:
            # 只显示选中的图表
            for viz_type, img_label in self.viz_images.items():
                title = self.get_viz_title(viz_type)
                img_label.parent().setVisible(title == selected)
    
    def refresh_visualization(self):
        """刷新可视化结果"""
        # 可能的可视化文件路径
        viz_paths = {
            "pareto": "solutions/scatter_matrix.png",  # 散点矩阵图
            "parallel": "solutions/parallel_coordinates.png",
            "radar": "solutions/radar_chart.png",
            "heatmap": "solutions/heatmap.png"
        }
        
        is_tsp = self.problem_type.currentText() == "tsp"
        if is_tsp:
            viz_paths = {
                "tsp_route": "solutions/tsp_solution.png",
                "tsp_front": "solutions/tsp_pareto_front.png",
            }
        for viz_type, label in self.viz_images.items():
            label.clear()
            label.parent().setVisible(viz_type in viz_paths)

        # 添加调试信息
        print(f"正在刷新可视化，当前工作目录: {os.getcwd()}")
        print(f"查找以下文件: {list(viz_paths.values())}")
        
        found_any = False
        
        # 检查文件是否存在并加载
        for viz_type, path in viz_paths.items():
            try:
                # 使用绝对路径
                abs_path = os.path.join(os.getcwd(), path)
                if os.path.exists(abs_path):
                    print(f"找到文件: {abs_path}")
                    file_size = os.path.getsize(abs_path)
                    print(f"文件大小: {file_size} 字节")
                    
                    pixmap = QPixmap(abs_path)
                    if not pixmap.isNull():
                        print(f"成功加载图像: {viz_type}, 大小: {pixmap.width()}x{pixmap.height()}")
                        if viz_type in self.viz_images:
                            self.viz_images[viz_type].setPixmap(pixmap.scaled(
                                self.viz_images[viz_type].width(), 
                                self.viz_images[viz_type].height(),
                                Qt.KeepAspectRatio, Qt.SmoothTransformation))
                            found_any = True
                    else:
                        print(f"无法加载图像: {abs_path}, pixmap为空")
                else:
                    print(f"文件不存在: {abs_path}")
            except Exception as e:
                print(f"加载可视化图像时出错: {viz_type} - {path}: {str(e)}")
        
        # 如果没有找到可视化结果，检查是否有特定图片，但不要加载high_dimensional_solution.png
        if not found_any and not is_tsp:
            print("没有找到标准可视化图像，尝试查找通用图像...")
            generic_paths = [
                "solutions/latest_solution.png",
                "visualizations/rl_solution.png"
            ]
            
            for path in generic_paths:
                try:
                    abs_path = os.path.join(os.getcwd(), path)
                    if os.path.exists(abs_path):
                        print(f"找到通用图像: {abs_path}")
                        pixmap = QPixmap(abs_path)
                        if not pixmap.isNull():
                            # 将通用图像只分配给Pareto前沿标签
                            if "pareto" in self.viz_images:
                                self.viz_images["pareto"].setPixmap(pixmap.scaled(
                                    self.viz_images["pareto"].width(), 
                                    self.viz_images["pareto"].height(),
                                    Qt.KeepAspectRatio, Qt.SmoothTransformation))
                                found_any = True
                                break
                        else:
                            print(f"无法加载通用图像: {abs_path}, pixmap为空")
                    else:
                        print(f"通用图像文件不存在: {abs_path}")
                except Exception as e:
                    print(f"加载通用图像时出错: {path}: {str(e)}")
        
        if found_any:
            self.statusBar.showMessage("已加载可视化结果")
            print("成功加载可视化结果")
        else:
            self.statusBar.showMessage("未找到可视化结果")
            print("警告：未找到任何可视化结果！")
            
            # 尝试列出solutions目录内容
            try:
                solutions_dir = os.path.join(os.getcwd(), "solutions")
                if os.path.exists(solutions_dir) and os.path.isdir(solutions_dir):
                    print(f"solutions目录内容:")
                    for file in os.listdir(solutions_dir):
                        print(f" - {file}")
                else:
                    print(f"solutions目录不存在!")
            except Exception as e:
                print(f"列出solutions目录内容时出错: {str(e)}")
    
    def refresh_model_list(self):
        """刷新模型列表（已禁用）"""
        # 由于我们已移除模型管理模块，此方法仅作为兼容性保留
        # 不再执行任何操作
        pass
    
    def delete_model(self):
        selected_items = self.model_list.selectedItems()
        if not selected_items:
            return
        
        for item in selected_items:
            text = item.text()
            model_path = text.split('(')[-1].rstrip(')')
            
            reply = QMessageBox.question(
                self, "确认删除",
                f"确定要删除模型 {model_path} 吗？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                try:
                    os.remove(model_path)
                    self.statusBar.showMessage(f"已删除模型: {model_path}")
                    self.refresh_model_list()
                except Exception as e:
                    QMessageBox.critical(self, "删除错误", f"删除模型时出错: {str(e)}")
    
    def use_selected_model(self):
        selected_items = self.model_list.selectedItems()
        if not selected_items:
            return
        
        item = selected_items[0]
        text = item.text()
        model_path = text.split('(')[-1].rstrip(')')
        
        self.checkpoint_path.setText(model_path)
        self.run_optimization("test")
        self.statusBar.showMessage(f"已选择模型: {model_path}")
    
    def load_generated_data(self):
        """加载生成的数据到表格中"""
        try:
            # 获取当前问题类型
            problem_type = self.problem_type.currentText().lower()
            
            # 获取最新生成的数据文件
            data_dir = os.path.join('data', problem_type, 'train')
            if not os.path.exists(data_dir):
                return
                
            files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
            if not files:
                return
                
            latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(data_dir, f)))
            file_path = os.path.join(data_dir, latest_file)
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if problem_type == 'knapsack':
                # 背包问题数据处理
                self._load_knapsack_data(data)
            elif problem_type == 'tsp':
                # TSP问题数据处理
                self._load_tsp_data(data)
            else:
                self.log_output.append(f"\n不支持的问题类型: {problem_type}")
            
        except Exception as e:
            if hasattr(self, 'log_output'):
                self.log_output.append(f"\n加载数据时出错: {str(e)}")
            print(f"加载数据时出错: {str(e)}")
    
    def _load_knapsack_data(self, data):
        """加载背包问题数据"""
        items = np.array(data['items'])
        initial_solution = np.array(data.get('initial_solution', []))
        
        # 获取目标维度
        n_objectives = items.shape[2]
        
        # 设置表格列数
        self.data_table.setColumnCount(n_objectives + 3)  # ID, 类别, 目标1...N, 是否选中
        
        # 创建列标题
        column_headers = ['ID', '类别']
        objective_names = data.get('objective_names', [f"目标{i+1}" for i in range(n_objectives)])
        column_headers.extend(objective_names)
        column_headers.append('是否选中')
        self.data_table.setHorizontalHeaderLabels(column_headers)
        
        # 设置表格行数
        total_items = items.shape[0] * items.shape[1]
        self.data_table.setRowCount(total_items)
        
        # 填充数据
        row = 0
        for cat_idx in range(items.shape[0]):
            for item_idx in range(items.shape[1]):
                self.data_table.setItem(row, 0, QTableWidgetItem(str(row)))
                self.data_table.setItem(row, 1, QTableWidgetItem(str(cat_idx)))
                
                # 填充所有目标属性
                for obj_idx in range(n_objectives):
                    value = items[cat_idx, item_idx, obj_idx]
                    self.data_table.setItem(row, 2 + obj_idx, QTableWidgetItem(f"{value:.2f}"))
                
                # 是否选中
                is_selected = "是" if len(initial_solution) > cat_idx and initial_solution[cat_idx] == item_idx else "否"
                self.data_table.setItem(row, n_objectives + 2, QTableWidgetItem(is_selected))
                row += 1
        
        # 调整列宽
        self.data_table.resizeColumnsToContents()
        
        # 显示约束信息
        if hasattr(self, 'log_output'):
            self.log_output.append("\n容量约束信息:")
            for i, name in enumerate(objective_names):
                if i > 0:  # 跳过第一个目标（通常是价值）
                    capacity_key = f"{name}_capacity"
                    capacity_value = data.get(capacity_key, data.get(f"{name.lower()}_capacity", "N/A"))
                    self.log_output.append(f"{name}容量: {capacity_value}")
        
        self.log_output.append(f"\n加载背包问题数据，目标数量: {n_objectives}")
        self.log_output.append(f"目标名称: {objective_names}")
    
    def _load_tsp_data(self, data):
        """加载TSP问题数据"""
        # 获取城市坐标和距离矩阵
        coordinates = np.array(data['coordinates'])
        distance_matrix = np.array(data['distance_matrix'])
        
        # 获取目标函数相关信息
        objective_names = data.get('objective_names', ["总距离", "时间窗口", "平衡性"])
        num_objectives = data.get('num_objectives', len(objective_names))
        
        # 设置表格列数和列标题 
        if 'nearest_neighbor_tour' in data:
            # 如果数据中包含最近邻解，显示解信息
            self.data_table.setColumnCount(5)  # ID, X坐标, Y坐标, 顺序, 到下一城市距离
            column_headers = ['城市ID', 'X坐标', 'Y坐标', '访问顺序', '到下一城市距离']
            
            # 设置表格行数
            self.data_table.setRowCount(len(coordinates))
            
            # 填充数据
            tour = np.array(data['nearest_neighbor_tour'])
            
            # 创建城市ID到访问顺序的映射
            tour_order = {city: idx for idx, city in enumerate(tour)}
            
            for i in range(len(coordinates)):
                # 城市ID
                self.data_table.setItem(i, 0, QTableWidgetItem(str(i)))
                
                # X、Y坐标
                self.data_table.setItem(i, 1, QTableWidgetItem(f"{coordinates[i][0]:.4f}"))
                self.data_table.setItem(i, 2, QTableWidgetItem(f"{coordinates[i][1]:.4f}"))
                
                # 访问顺序
                order = tour_order.get(i, -1)
                self.data_table.setItem(i, 3, QTableWidgetItem(str(order)))
                
                # 到下一城市的距离
                if i in tour[:-1]:  # 排除最后一个（返回起点）
                    idx = np.where(tour == i)[0][0]
                    next_city = tour[idx + 1]
                    distance = distance_matrix[i][next_city]
                    self.data_table.setItem(i, 4, QTableWidgetItem(f"{distance:.4f}"))
                else:
                    self.data_table.setItem(i, 4, QTableWidgetItem("N/A"))
        else:
            # 如果没有解信息，只显示坐标
            self.data_table.setColumnCount(3)  # ID, X坐标, Y坐标
            column_headers = ['城市ID', 'X坐标', 'Y坐标']
            
            # 设置表格行数
            self.data_table.setRowCount(len(coordinates))
            
            # 填充数据
            for i in range(len(coordinates)):
                self.data_table.setItem(i, 0, QTableWidgetItem(str(i)))
                self.data_table.setItem(i, 1, QTableWidgetItem(f"{coordinates[i][0]:.4f}"))
                self.data_table.setItem(i, 2, QTableWidgetItem(f"{coordinates[i][1]:.4f}"))
        
        # 设置列标题
        self.data_table.setHorizontalHeaderLabels(column_headers)
        
        # 调整列宽
        self.data_table.resizeColumnsToContents()
        
        # 在日志中显示目标函数信息
        self.log_output.append(f"\n加载TSP问题数据，城市数量: {len(coordinates)}")
        self.log_output.append(f"目标数量: {num_objectives}")
        self.log_output.append(f"目标名称: {objective_names}")
        
        # 在日志中显示距离矩阵摘要
        self.log_output.append(f"\n距离矩阵摘要:")
        self.log_output.append(f"最小距离: {distance_matrix.min():.4f}")
        self.log_output.append(f"最大距离: {distance_matrix.max():.4f}")
        self.log_output.append(f"平均距离: {distance_matrix.mean():.4f}")
    
    def export_data(self):
        """导出表格数据到CSV文件"""
        try:
            # 获取保存文件路径
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "导出数据",
                "",
                "CSV文件 (*.csv);;Excel文件 (*.xlsx)"
            )
            
            if not file_path:
                return
                
            if file_path.endswith('.csv'):
                # 导出为CSV
                with open(file_path, 'w', encoding='utf-8-sig') as f:
                    # 写入表头
                    headers = []
                    for col in range(self.data_table.columnCount()):
                        headers.append(self.data_table.horizontalHeaderItem(col).text())
                    f.write(','.join(headers) + '\n')
                    
                    # 写入数据
                    for row in range(self.data_table.rowCount()):
                        row_data = []
                        for col in range(self.data_table.columnCount()):
                            item = self.data_table.item(row, col)
                            if item is not None:
                                row_data.append(item.text().replace(',', ';'))
                            else:
                                row_data.append('')
                        f.write(','.join(row_data) + '\n')
                        
            elif file_path.endswith('.xlsx'):
                try:
                    import pandas as pd
                    
                    # 收集数据
                    data = []
                    headers = []
                    
                    # 获取表头
                    for col in range(self.data_table.columnCount()):
                        headers.append(self.data_table.horizontalHeaderItem(col).text())
                    
                    # 获取数据
                    for row in range(self.data_table.rowCount()):
                        row_data = []
                        for col in range(self.data_table.columnCount()):
                            item = self.data_table.item(row, col)
                            if item is not None:
                                row_data.append(item.text())
                            else:
                                row_data.append('')
                        data.append(row_data)
                    
                    # 创建DataFrame并保存
                    df = pd.DataFrame(data, columns=headers)
                    df.to_excel(file_path, index=False)
                    
                except ImportError:
                    QMessageBox.warning(self, "警告", "导出Excel需要安装pandas库，将改为导出CSV格式")
                    # 递归调用，改为保存CSV
                    self.export_data()
                    return
            
            self.statusBar.showMessage(f"数据已导出到: {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "导出错误", f"导出数据时出错: {str(e)}")
            self.statusBar.showMessage("导出失败")
    
    def run_optimization(self, mode):
        """运行优化"""
        # 获取问题类型
        problem_type = self.problem_type.currentText().lower()
        
        # 获取目标设置
        objectives = []
        objective_types = []
        objective_names = []
        objective_english_names = []  # 添加英文名称列表
        objective_ranges = []
        objective_constraints = {}
        
        for i in range(self.objectives_table.rowCount()):
            name = self.objectives_table.item(i, 0).text()
            # 为常见中文目标名称提供英文映射
            english_name = name
            if name == "价值":
                english_name = "Value"
            elif name == "重量":
                english_name = "Weight"
            elif name == "体积":
                english_name = "Volume"
            elif name == "成本":
                english_name = "Cost"
            
            type_combo = self.objectives_table.cellWidget(i, 1)
            type_value = 1 if type_combo.currentText() == "最大化" else -1
            
            constraint_combo = self.objectives_table.cellWidget(i, 2)
            is_constraint = constraint_combo.currentText() == "是"
            
            constraint_value = self.objectives_table.cellWidget(i, 3)
            constraint_val = float(constraint_value.text()) if is_constraint and constraint_value.text() != "N/A" else None
            
            min_value = float(self.objectives_table.cellWidget(i, 4).text())
            max_value = float(self.objectives_table.cellWidget(i, 5).text())
            
            objectives.append({
                "name": name,
                "type": type_value,
                "is_constraint": is_constraint,
                "constraint_value": constraint_val,
                "range": [min_value, max_value]
            })
            objective_types.append(type_value)
            objective_names.append(name)
            objective_english_names.append(english_name)
            objective_ranges.append([min_value, max_value])
            if is_constraint and constraint_val is not None:
                objective_constraints[name] = constraint_val
        
        # 获取模型参数
        hidden_dim = self.hidden_dim.value()
        
        # 获取检查点路径
        checkpoint_path = self.checkpoint_path.text() if mode in ["test", "train"] else None
        
        # 构建命令行参数
        cmd = [
            "python", "scripts/main.py",
            "--mode", mode,
            "--problem", problem_type,
            "--rl",
            "--n-objectives", str(len(objectives))
        ]
        
        # 添加目标名称，确保正确处理
        if objective_english_names:
            objective_names_str = ",".join(objective_english_names)
            cmd.extend(["--objective-names", objective_names_str])
        
        # 添加目标类型，确保正确处理
        if objective_types:
            objective_types_str = ",".join(map(str, objective_types))
            # 在Windows下，使用引号包裹以避免命令行解析问题
            if sys.platform == 'win32':
                objective_types_str = f'"{objective_types_str}"'
            cmd.extend(["--objective-types", objective_types_str])
        
        if checkpoint_path:
            cmd.extend(["--checkpoint", checkpoint_path])
        
        if mode == "test":
            cmd.append("--visualize")
        
        # 创建临时配置文件
        num_categories = self.num_categories.value()
        items_per_category = self.items_per_category.value()
        
        # 根据问题类型构建配置
        if problem_type == "knapsack":
            problem_config = {
                "type": problem_type,
                "num_categories": num_categories,
                "items_per_category": items_per_category,
                "n_objectives": len(objectives),
                "objective_names": objective_names,
                "objective_types": objective_types,
                "objective_ranges": objective_ranges,
                "objective_constraints": objective_constraints
            }
            
            # 其他问题类型的默认模型配置
            model_config = {
                "hidden_dim": hidden_dim,
                "input_dim": len(objectives) + 1  # 动态设置：属性数量 + 类别信息
            }
        elif problem_type == "tsp":
            # TSP问题的配置
            problem_config = {
                "type": problem_type,
                "size": num_categories,  # 对于TSP，使用类别数量作为城市数量
                "n_objectives": len(objectives),
                "objective_names": objective_names,
                "objective_types": objective_types
            }
            
            # 为TSP问题设置特定的模型参数
            model_config = {
                "hidden_dim": hidden_dim,
                "input_dim": 4,  # TSP的固定输入维度：[相对x, 相对y, 距离, 已访问比例]
                "n_objectives": len(objectives)
            }
        else:
            problem_config = {
                "type": problem_type,
                "size": num_categories * items_per_category,
                "n_objectives": len(objectives),
                "objective_names": objective_names,
                "objective_types": objective_types
            }
            
            # 其他问题类型的默认模型配置
            model_config = {
                "hidden_dim": hidden_dim,
                "input_dim": len(objectives) + 1,  # 动态设置：属性数量 + 类别信息
                "n_objectives": len(objectives)
            }
        
        config = {
            "problem": problem_config,
            "model": model_config,
            "training": {
                "epochs": 100,
                "batch_size": 32,
                "learning_rate": 0.001,
                "num_episodes": 1000,
                "save_interval": 50,
                "eval_interval": 100,
                "max_steps": 100,
                "gamma": 0.99,
                "epsilon_start": 1.0,
                "epsilon_end": 0.01,
                "epsilon_decay": 0.995,
                "buffer_size": 10000,
                "entropy_weight": self.entropy_weight.value(),
                "value_weight": self.value_weight.value(),
                "checkpoint_dir": os.path.join("checkpoints", problem_type)
            },
            "data": {
                "train_dir": os.path.join("data", problem_type, "train"),
                "val_dir": os.path.join("data", problem_type, "val"),
                "test_dir": os.path.join("data", problem_type, "test"),
                "train_size": 1000,  # 训练集大小
                "val_size": 200,     # 验证集大小
                "test_size": 100     # 测试集大小
            },
            "visualization": {
                "use_english_labels": True,
                "dpi": 150,  # 设置更高的DPI以获得更清晰的图像
                "style": "seaborn-v0_8-whitegrid"  # 使用更现代的样式
            }
        }
        
        # 确保配置目录存在
        os.makedirs("configs", exist_ok=True)
        config_path = os.path.join("configs", "temp_config.yaml")
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True)
        
        cmd.extend(["--config-file", config_path])
        
        # 更新状态栏
        self.statusBar.showMessage(f"正在准备执行: {' '.join(cmd)}")
        
        # 在日志区域显示命令和配置
        self.log_output.clear()
        self.log_output.append(f"执行命令: {' '.join(cmd)}\n")
        self.log_output.append(f"配置文件内容:\n{yaml.dump(config, allow_unicode=True)}\n")
        
        # 在日志中添加警告
        if len(objectives) > 4 and mode == "generate":
            self.log_output.append("\n⚠️ 警告: 检测到超过4个目标。后端数据生成器可能存在硬编码限制，只会生成4个目标的数据。\n" + 
                                  "需要修改后端代码(可能是scripts/generate_data.py)以支持更多目标。")
        
        # 在日志中添加调试信息
        if mode == "train":
            self.log_output.append(f"\n调试信息：模型输入维度已设置为 {len(objectives) + 1}")
        
        # 创建并启动线程而不是直接使用subprocess
        self.run_thread = RunThread(cmd)
        self.run_thread.output_signal.connect(self.update_log)
        self.run_thread.finished_signal.connect(lambda exit_code: self.process_finished(exit_code, mode))
        self.run_thread.start()
        
        # 禁用按钮，防止多次点击
        self.disable_operation_buttons()

    def update_log(self, text):
        """更新日志输出"""
        # 添加到内存缓冲区
        self.log_buffer.append(text)
        
        # 记录到文件
        self.file_logger.info(text)
        
        # 如果缓冲区过大，立即刷新
        if len(self.log_buffer) > 200:
            self.flush_log_buffer()
    
    def flush_log_buffer(self):
        """将日志缓冲区内容写入UI"""
        if not self.log_buffer:
            return
            
        # 批量处理所有缓冲的日志
        buffer_text = "\n".join(self.log_buffer)
        self.log_buffer = []
        
        # 限制日志区域的内容长度，防止内存溢出
        current_text = self.log_output.toPlainText()
        if len(current_text) > 100000:  # 限制日志长度为10万字符
            # 保留最新的5万字符
            self.log_output.setPlainText(current_text[-50000:] + "\n--- 日志已截断，仅显示最新部分 ---\n")
        
        # 添加新日志
        self.log_output.append(buffer_text)
        
        # 滚动到底部
        cursor = self.log_output.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_output.setTextCursor(cursor)

    def process_finished(self, exit_code, mode):
        """处理命令执行完成的回调"""
        # 更新状态栏和记录结束状态
        end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.log_output.append(f"\n[{end_time}] 执行完成，退出代码: {exit_code}")
        
        status = "完成" if exit_code == 0 else f"失败 (错误代码: {exit_code})"
        self.statusBar.showMessage(f"命令执行{status}")
        
        # 清理临时配置文件
        config_path = os.path.join("configs", "temp_config.yaml")
        if os.path.exists(config_path):
            try:
                os.remove(config_path)
            except Exception as e:
                self.log_output.append(f"警告：清理临时配置文件时出错: {str(e)}")
        
        # 根据模式执行后续操作
        if exit_code == 0:
            if mode == "generate":
                # 加载生成的数据
                self.load_generated_data()
                # 切换到数据查看选项卡
                self.switch_to_tab("数据查看")
                self.log_output.append("\n数据生成成功。已加载生成的数据。")
            elif mode == "train":
                # 显示训练完成
                self.log_output.append("\n模型训练成功。查看checkpoint目录获取保存的模型。")
                
                # 弹出窗口显示训练损失曲线
                try:
                    loss_image_path = os.path.join('solutions', 'latest_training_loss.png')
                    if os.path.exists(loss_image_path):
                        self.show_training_loss_popup(loss_image_path)
                    else:
                        self.log_output.append("\n警告：找不到训练损失曲线图像。")
                except Exception as e:
                    self.log_output.append(f"\n显示训练损失弹窗时出错: {str(e)}")
                
            elif mode == "test":
                # 显示测试完成，刷新可视化
                self.log_output.append("\n模型测试成功。查看solutions目录获取可视化结果。")
                # 切换到可视化选项卡
                self.switch_to_tab("可视化")
                
                # 在测试完成后强制刷新可视化
                print("\n尝试刷新可视化...")
                self.refresh_visualization()
        else:
            # 测试失败也尝试刷新可视化
            if mode == "test":
                self.log_output.append("\n模型测试失败，但仍尝试加载可能生成的可视化结果。")
                print("\n测试失败，但仍尝试刷新可视化...")
                self.switch_to_tab("可视化")
                self.refresh_visualization()
        
        # 重新启用按钮
        self.enable_operation_buttons()

    def show_training_loss_popup(self, image_path):
        """显示训练损失曲线弹出窗口"""
        try:
            # 创建对话框
            dialog = QDialog(self)
            dialog.setWindowTitle("训练损失曲线")
            dialog.setMinimumSize(800, 600)
            
            # 创建布局
            layout = QVBoxLayout(dialog)
            
            # 创建图片标签
            image_label = QLabel()
            pixmap = QPixmap(image_path)
            
            # 如果图片太大，适当缩小
            if pixmap.width() > 780 or pixmap.height() > 580:
                pixmap = pixmap.scaled(780, 580, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                
            image_label.setPixmap(pixmap)
            image_label.setAlignment(Qt.AlignCenter)
            
            # 添加图片标签到布局
            layout.addWidget(image_label)
            
            # 添加关闭按钮
            btn_layout = QHBoxLayout()
            close_btn = QPushButton("关闭")
            close_btn.clicked.connect(dialog.accept)
            btn_layout.addStretch()
            btn_layout.addWidget(close_btn)
            btn_layout.addStretch()
            layout.addLayout(btn_layout)
            
            # 显示对话框
            dialog.exec_()
        except Exception as e:
            self.log_output.append(f"创建训练损失弹窗出错: {str(e)}")

    def switch_to_tab(self, tab_name):
        """切换到指定名称的选项卡"""
        for i in range(self.main_layout.count()):
            widget = self.main_layout.itemAt(i).widget()
            if isinstance(widget, QTabWidget):
                for j in range(widget.count()):
                    if widget.tabText(j) == tab_name:
                        widget.setCurrentIndex(j)
                        break

    def disable_operation_buttons(self):
        """禁用操作按钮"""
        for child in self.findChildren(QPushButton):
            if child.text() in ["生成数据", "训练模型", "测试模型"]:
                child.setEnabled(False)
        
        if hasattr(self, 'cancel_button'):
            self.cancel_button.setEnabled(True)

    def enable_operation_buttons(self):
        """启用操作按钮"""
        for child in self.findChildren(QPushButton):
            if child.text() in ["生成数据", "训练模型", "测试模型"]:
                child.setEnabled(True)
        
        if hasattr(self, 'cancel_button'):
            self.cancel_button.setEnabled(False)

    def cancel_operation(self):
        """取消当前运行的操作"""
        if hasattr(self, 'run_thread') and self.run_thread.isRunning():
            self.log_output.append("\n取消中... 请稍等。程序可能需要一段时间才能响应。")
            self.run_thread.stop()
            self.enable_operation_buttons()
            self.statusBar.showMessage("操作已取消")
    
    def problem_type_changed(self, problem_type):
        """问题类型变更时的处理"""
        # 更新数据类型指示标签
        if hasattr(self, 'data_type_label'):
            if problem_type == "knapsack":
                self.data_type_label.setText("当前数据类型: 背包问题")
            elif problem_type == "tsp":
                self.data_type_label.setText("当前数据类型: 旅行商问题")
            else:
                self.data_type_label.setText(f"当前数据类型: {problem_type}")
        
        is_tsp = problem_type == "tsp"
        self.num_categories_label.setText("城市数量:" if is_tsp else "类别数量:")
        self.items_per_category_label.setText("(不适用)" if is_tsp else "每类物品数:")
        self.items_per_category.setEnabled(not is_tsp)
        objective_range = (1, len(TSP_OBJECTIVE_NAMES)) if is_tsp else (4, 10)
        self.n_objectives.setRange(*objective_range)
        self.update_objectives_table()
        self.refresh_visualization()

        # 清空数据表格
        self.data_table.clearContents()
        self.data_table.setRowCount(0)
        
        # 尝试加载最新数据
        try:
            # 检查是否存在数据文件
            data_dir = os.path.join('data', problem_type, 'train')
            if os.path.exists(data_dir):
                files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
                if files:
                    # 有数据文件，加载最新的
                    self.log_output.append(f"\n检测到{problem_type}问题的数据文件，尝试加载...")
                    self.load_generated_data()
                else:
                    # 没有数据文件，设置相应的表头
                    if problem_type == 'knapsack':
                        self.data_table.setColumnCount(7)
                        self.data_table.setHorizontalHeaderLabels(['ID', '类别', '价值', '重量', '体积', '成本', '是否选中'])
                    elif problem_type == 'tsp':
                        self.data_table.setColumnCount(3)
                        self.data_table.setHorizontalHeaderLabels(['城市ID', 'X坐标', 'Y坐标'])
            else:
                # 数据目录不存在，设置默认表头
                if problem_type == 'knapsack':
                    self.data_table.setColumnCount(7)
                    self.data_table.setHorizontalHeaderLabels(['ID', '类别', '价值', '重量', '体积', '成本', '是否选中'])
                elif problem_type == 'tsp':
                    self.data_table.setColumnCount(3)
                    self.data_table.setHorizontalHeaderLabels(['城市ID', 'X坐标', 'Y坐标'])
        except Exception as e:
            self.log_output.append(f"\n尝试加载数据时出错: {str(e)}")
            # 出错时设置默认表头
            if problem_type == 'knapsack':
                self.data_table.setColumnCount(7)
                self.data_table.setHorizontalHeaderLabels(['ID', '类别', '价值', '重量', '体积', '成本', '是否选中'])
            elif problem_type == 'tsp':
                self.data_table.setColumnCount(3)
                self.data_table.setHorizontalHeaderLabels(['城市ID', 'X坐标', 'Y坐标'])
        
        # 记录日志
        self.log_output.append(f"\n切换到{problem_type}问题类型")
        self.statusBar.showMessage(f"已切换到{problem_type}问题类型")

    def closeEvent(self, event):
        """应用程序关闭时的事件处理"""
        # 询问用户是否确认退出
        if hasattr(self, 'run_thread') and self.run_thread.isRunning():
            reply = QMessageBox.question(
                self, "确认退出", 
                "正在运行训练进程，确定要退出应用程序吗？\n这将终止所有正在运行的训练。",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                event.ignore()
                return
                
            # 用户确认退出，终止所有进程
            self.cancel_operation()
            
        # 关闭日志计时器
        if hasattr(self, 'log_timer') and self.log_timer.isActive():
            self.log_timer.stop()
            
        # 记录关闭信息到日志
        if hasattr(self, 'file_logger'):
            self.file_logger.info("GUI会话已结束")
            
        # 接受关闭事件
        event.accept()


class RunThread(QThread):
    output_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(int)
    buffer_signal = pyqtSignal()  # 添加刷新缓冲区的信号
    
    def __init__(self, cmd):
        super().__init__()
        self.cmd = cmd
        self.process = None
        self.buffer = []
        self.buffer_limit = 100  # 缓冲区大小限制
        self.running = True
        
        # 连接缓冲区信号到刷新方法
        self.buffer_signal.connect(self.flush_buffer)
    
    def run(self):
        try:
            self.process = subprocess.Popen(
                self.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # 创建一个单独的线程来定期刷新缓冲区
            import threading
            buffer_thread = threading.Thread(target=self.periodic_flush)
            buffer_thread.daemon = True  # 设置为守护线程，主线程结束时会自动结束
            buffer_thread.start()
            
            # 读取输出并缓存
            for line in iter(self.process.stdout.readline, ''):
                if not self.running:
                    break
                    
                if line:
                    # 线程安全地添加到缓冲区
                    with threading.Lock():
                        self.buffer.append(line.strip())
                    # 当缓冲区达到限制时，发送信号强制刷新
                    if len(self.buffer) >= self.buffer_limit:
                        self.buffer_signal.emit()
            
            # 最后一次刷新缓冲区
            self.buffer_signal.emit()
            
            # 等待进程完成并发送退出代码
            exit_code = self.process.wait()
            self.running = False
            self.finished_signal.emit(exit_code)
                
        except Exception as e:
            self.output_signal.emit(f"错误: {str(e)}")
            self.finished_signal.emit(1)
        finally:
            self.running = False
            # 确保进程资源被释放
            if self.process:
                try:
                    self.process.stdout.close()
                except:
                    pass
    
    def periodic_flush(self):
        """定期刷新缓冲区的方法，在一个单独的线程中运行"""
        import time
        while self.running:
            time.sleep(0.5)  # 每500毫秒
            if self.buffer:
                self.buffer_signal.emit()
                    
    def flush_buffer(self):
        """将缓冲区的内容发送到UI"""
        import threading
        if self.buffer:
            with threading.Lock():
                # 每次最多发送50行，避免UI过载
                batch_size = min(50, len(self.buffer))
                output_batch = "\n".join(self.buffer[:batch_size])
                # 移除已发送的内容
                self.buffer = self.buffer[batch_size:]
            
            # 发送内容到UI
            self.output_signal.emit(output_batch)
    
    def stop(self):
        """停止正在运行的进程"""
        self.running = False
            
        if self.process and self.process.poll() is None:
            try:
                if sys.platform == 'win32':
                    # Windows下使用taskkill强制结束进程
                    subprocess.call(['taskkill', '/F', '/T', '/PID', str(self.process.pid)])
                else:
                    # Linux/Mac下使用terminate
                    self.process.terminate()
                    self.process.wait(timeout=3)
            except:
                # 如果终止失败，尝试强制终止
                try:
                    self.process.kill()
                except:
                    pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MultiObjectiveOptimizer()
    window.show()
    sys.exit(app.exec_()) 