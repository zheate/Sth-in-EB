from functools import lru_cache


@lru_cache
def cross_point_xy(line1: tuple, line2: tuple):  # 计算交点函数
    x1 = line1[0][0]  # 取直线1的第一个点坐标
    y1 = line1[0][1]
    x2 = line1[1][0]  # 取直线1的第二个点坐标
    y2 = line1[1][1]

    x3 = line2[0][0]  # 取直线2的第一个点坐标
    y3 = line2[0][1]
    x4 = line2[1][0]  # 取直线2的第二个点坐标
    y4 = line2[1][1]

    if x1 == x2:  # L1 直线斜率不存在
        k1 = None
        b1 = None
    else:
        k1 = (y2 - y1) / (x2 - x1)
        b1 = y1 - x1 * k1

    if x3 == x4:  # L2直线斜率不存在操作
        k2 = None
        b2 = None
    else:
        k2 = (y4 - y3) / (x4 - x3)
        b2 = y3 - x3 * k2

    if k1 is None and k2 is None:  # L1与L2直线斜率都不存在，两条直线均与y轴平行
        return None
    elif k1 is not None and k2 is None:  # 若L2与y轴平行，L1为一般直线，交点横坐标为L2的x坐标
        x = x3
        y = k1 * x + b1
    elif k1 is None and k2 is not None:  # 若L1与y轴平行，L2为一般直线，交点横坐标为L1的x坐标
        x = x1
        y = k2 * x + b2
    else:  # 两条一般直线
        if k1 == k2:  # 两直线斜率相同
            return None
        else:  # 两直线不平行，必然存在交点
            x = (b2 - b1) / (k1 - k2)
            y = k1 * x + b1
    return x, y


@lru_cache
def cross_point_kb(line1: tuple, line2: tuple):
    k1 = line1[0]  # 取直线1的kb
    b1 = line1[1]

    k2 = line2[0]  # 取直线2的kb
    b2 = line2[1]

    if k1 is None and k2 is None:  # L1与L2直线斜率都不存在，两条直线均与y轴平行
        return None
    elif k1 is not None and k2 is None:  # 若L2与y轴平行，L1为一般直线，交点横坐标为L2的x坐标
        x = b2
        y = k1 * x + b1
    elif k1 is None and k2 is not None:  # 若L1与y轴平行，L2为一般直线，交点横坐标为L1的x坐标
        x = b1
        y = k2 * x + b2
    else:  # 两条一般直线
        if k1 == k2:  # 两直线斜率相同
            return None
        else:  # 两直线不平行，必然存在交点
            x = (b2 - b1) / (k1 - k2)
            y = k1 * x + b1
    return x, y


if __name__ == '__main__':
    print(cross_point_xy(((1, 2), (3, 4)), ((1, 2), (2, 6))))
    print(cross_point_kb((None, 1), (None, 0)))
