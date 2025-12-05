def fiber_bend_na_calculate(original_na, fiber_core_diameter, fiber_core_index, blend_diameter):
    temp = (fiber_core_diameter / 2 * (1 + fiber_core_diameter / blend_diameter) /
            (blend_diameter / 2 * ((original_na / fiber_core_index) ** 2 -
                                   (2 + fiber_core_diameter / blend_diameter) * fiber_core_diameter / blend_diameter)))
    na = (1 - temp) * original_na
    return na


if __name__ == '__main__':
    from numpy import linspace
    from matplotlib.pyplot import plot, show, xlabel, ylabel, rcParams

    rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    a = 0.22
    b = 135e-6
    c = 1.4673105070761265
    d = linspace(50e-3, 240e-3, 100)

    e = fiber_bend_na_calculate(a, b, c, d)
    plot(d / 2 * 1000, e)
    xlabel('光纤盘绕半径/mm')
    ylabel('实际 NA')
    show(block=True)
