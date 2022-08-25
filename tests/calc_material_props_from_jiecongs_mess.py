import math

if __name__ == "__main__":
    # tube 2 outer
    EI_x = 0.008929444846481
    EI_y = 0.009185881509851
    d1 = 3.4   # mm
    d2 = 5.1 # mm
    d1 = d1/1000
    d2 = d2/1000
    r1 = d1/2
    r2 = d2/2

    Ix = (math.pi/4)*(r2**4 - r1**4)

    Ex = EI_x/Ix
    print("outer tube")
    print(f"Ex {Ex}")
    Ey = EI_y/Ix # Ix because Ix = Iy -> we assume a perfect annula
    print(f"Ey {Ey}")
    print()
    print(f"mean of prev. values {(Ey + Ex)/2}") # approximation
    print()
    GJ = 0.005217755559196 # Pa·m ^ 4
    J = math.pi*(d2**4 - d1**4)/32
    print(f"G {GJ/J}")
    print()
    print()

    # tube 1 inner
    EI_x = 0.001046249351773 #   Pa·m ^ 4 and
    EI_y = 0.001247095498471 #   Pa·m ^ 4;

    d1 = 0   # mm
    d2 = 2.8 # mm
    d2 = d2/1000
    r1 = d1/2
    r2 = d2/2

    Ix = (math.pi/4)*(r2**4 - r1**4)

    Ex = EI_x/Ix
    print("inner tube")
    print(f"Ex {Ex}")
    Ey = EI_y/Ix # Ix because Ix = Iy -> we assume a perfect annula
    print(f"Ey {Ey}")
    print()
    print(f"mean of prev. values {(Ey + Ex)/2}") # approximation
    GJ = 0.000644390038492 #Pa·m ^ 4
    J = math.pi*(d2**4 - d1**4)/32
    print(f"G {GJ/J}")


