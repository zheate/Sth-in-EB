# LD 光纤耦合模块设计

精简版项目，仅包含核心计算和可视化功能。

## 项目结构

```
LD/
├── streamlit_app.py          # Streamlit 主应用
├── requirements.txt          # Python 依赖
├── application/
│   ├── __init__.py
│   └── models/
│       ├── __init__.py
│       ├── cross_point.py           # 交叉点计算
│       ├── fiber_bend_na.py         # 光纤弯曲 NA 计算
│       ├── fresnel_equation.py      # 菲涅尔方程
│       ├── gaussian_beam.py         # 高斯光束模型
│       ├── laser_diode_calculation.py # 激光二极管主计算
│       ├── parallel_utils.py        # 并行计算工具
│       ├── parameters_conversion.py # 参数转换
│       └── width_calculate_one_dimension.py # 一维宽度计算
└── assets/
    └── ld_config.json        # JSON 配置文件
```

## 安装

```bash
pip install -r requirements.txt
```

## 运行

```bash
streamlit run streamlit_app.py
```

## 特性

- **JSON 配置**: 使用 JSON 格式存储参数，快速加载
- **并行计算**: 支持多核并行计算，优化性能
- **Plotly 可视化**: 交互式图表，平滑渲染
- **实时计算**: 参数修改后即时计算结果

