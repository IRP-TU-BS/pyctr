from manim import *
from cosserat_rod import *
from copy import copy
class RodPlot(Scene):
    def construct(self):
        ax = Axes(
            x_range=[-0.5, 0.5, 0.05],
            y_range=[-1, 5, 1],
            #x_length=9,
            #y_length=6,
            #x_axis_config={"numbers_to_include": np.arange(0, 40, 5)},
            #y_axis_config={"numbers_to_include": np.arange(-5, 34, 5)},
            tips=False,
        )
        labels = ax.get_axis_labels(
            x_label=Tex("x"), y_label=Tex("y")
        )

        rod = CosseratRod()
        # Arbitrary base frame assignment
        p0 = np.array([[0, 0, 0]])
        R0 = np.eye(3)
        rod.set_initial_conditions(p0, R0)

        for i, f in enumerate(np.linspace(0,-1,10)):
            wrench = np.array([f/10.,0.,f,0,0,0])
            states = rod.push_end(wrench)
            x_vals , y_vals = states[:,1], states[:,2]
            if i > 0:
                old_graph = copy(graph)
            graph = ax.plot_line_graph(x_values=x_vals, y_values=y_vals)
            if i == 0:
                self.play(Create(ax), Create(graph))#, Write(graph_label))
                self.wait()
            else:
                self.play(ReplacementTransform(old_graph, graph))
                self.wait()

class LineRodPlot(Scene):
    def construct(self):
        axes = Axes(
            x_range=[-3, 3],
            y_range=[-5, 5],
            axis_config={"color": BLUE},
        )

        # Create Graph
        graph = axes.get_graph(lambda x: x**2, color=WHITE)
        graph_label = axes.get_graph_label(graph, label='x^{2}')

        graph2 = axes.get_graph(lambda x: x**3, color=WHITE)
        graph_label2 = axes.get_graph_label(graph2, label='x^{3}')

        # Display graph
        self.play(Create(axes), Create(graph), Write(graph_label))
        self.wait(1)
        self.play(Transform(graph, graph2), Transform(graph_label, graph_label2))
        self.wait(1)
