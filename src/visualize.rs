use crate::NeuralNetwork;
use eframe::egui::{self, Color32};
use egui_graphs::{
    DefaultEdgeShape, DefaultNodeShape, Graph as EguiGraph, GraphView, Layout, LayoutState, SettingsNavigation,
    SettingsStyle,
};
use petgraph::stable_graph::NodeIndex;

// ── No-op layout: never moves nodes (we set positions manually) ────

#[derive(Clone, Default, Debug, serde::Serialize, serde::Deserialize)]
 struct FixedState;
impl LayoutState for FixedState {}
#[derive(Default)]
 struct FixedLayout;
impl Layout<FixedState> for FixedLayout {
    fn next<N, E, Ty, Ix, Dn, De>(
        &mut self,
        _g: &mut EguiGraph<N, E, Ty, Ix, Dn, De>,
        _ui: &egui::Ui,
    ) where
        N: Clone,
        E: Clone,
        Ty: petgraph::EdgeType,
        Ix: petgraph::graph::IndexType,
        Dn: egui_graphs::DisplayNode<N, E, Ty, Ix>,
        De: egui_graphs::DisplayEdge<N, E, Ty, Ix, Dn>,
    {
    }

    fn state(&self) -> FixedState {
        FixedState
    }
    fn from_state(_state: FixedState) -> impl Layout<FixedState> {
        FixedLayout
    }
}

// ── Types ─────────────────────────────────────────────────────────

type NetGraph = EguiGraph<
    (),
    (),
    petgraph::Directed,
    petgraph::stable_graph::DefaultIx,
    DefaultNodeShape,
    DefaultEdgeShape,
>;

type GraphViewWidget<'a> = GraphView<
    'a,
    (),
    (),
    petgraph::Directed,
    petgraph::stable_graph::DefaultIx,
    DefaultNodeShape,
    DefaultEdgeShape,
    FixedState,
    FixedLayout,
>;

struct LayerMeta {
    name: String,
    nodes: Vec<NodeIndex>,
    visible: bool,
}

// ── Colours ───────────────────────────────────────────────────────

const COLOR_INPUT: Color32 = Color32::from_rgb(70, 130, 220);
const COLOR_HIDDEN: Color32 = Color32::from_rgb(220, 180, 70);
const COLOR_OUTPUT: Color32 = Color32::from_rgb(70, 200, 120);

const DIM_INPUT: Color32 = Color32::from_rgb(30, 60, 110);
const DIM_HIDDEN: Color32 = Color32::from_rgb(110, 90, 35);
const DIM_OUTPUT: Color32 = Color32::from_rgb(35, 100, 60);

const HIDDEN: Color32 = Color32::from_rgb(20, 20, 20);

// ── Graph builder ─────────────────────────────────────────────────

fn build_graph_and_meta(net: &NeuralNetwork) -> (NetGraph, Vec<LayerMeta>) {
    let mut g = NetGraph::new(Default::default());
    let n_inputs = net.layers[0].neurons[0].input.len();

    let x_input = 0.0_f32;
    let x_spacing = 300.0_f32;
    let y_spacing = 80.0_f32;

    let mut layers: Vec<LayerMeta> = Vec::new();

    // Input layer
    {
        let mut nodes = Vec::with_capacity(n_inputs);
        let y_start = -(n_inputs as f32 - 1.0) * y_spacing / 2.0;
        for i in 0..n_inputs {
            let y = y_start + i as f32 * y_spacing;
            let idx = g.add_node_with_label_and_location(
                (),
                format!("Input[{i}]"),
                egui::Pos2::new(x_input, y),
            );
            g.g_mut()
                .node_weight_mut(idx)
                .unwrap()
                .set_color(COLOR_INPUT);
            nodes.push(idx);
        }
        layers.push(LayerMeta {
            name: format!("Input Layer ({n_inputs})"),
            nodes,
            visible: true,
        });
    }

    // Hidden & output layers
    for (layer_idx, layer) in net.layers.iter().enumerate() {
        let x = x_input + (layer_idx as f32 + 1.0) * x_spacing;
        let n = layer.neurons.len();
        let mut nodes = Vec::with_capacity(n);
        let y_start = -(n as f32 - 1.0) * y_spacing / 2.0;
        let is_last = layer_idx == net.layers.len() - 1;
        let (base_color, prefix): (Color32, String) = if is_last {
            (COLOR_OUTPUT, "Output".into())
        } else {
            (COLOR_HIDDEN, format!("Layer {}", layer_idx + 1))
        };

        for (neuron_idx, neuron) in layer.neurons.iter().enumerate() {
            let y = y_start + neuron_idx as f32 * y_spacing;
            let label = format!("{}[{}]  b={:.3}", prefix, neuron_idx, neuron.bias);
            let idx = g.add_node_with_label_and_location((), label, egui::Pos2::new(x, y));
            g.g_mut()
                .node_weight_mut(idx)
                .unwrap()
                .set_color(base_color);
            nodes.push(idx);
        }
        layers.push(LayerMeta {
            name: if is_last {
                format!("Output Layer ({n})")
            } else {
                format!("Layer {} ({n})", layer_idx + 1)
            },
            nodes,
            visible: true,
        });
    }

    // Edges — connect each layer to the next
    for i in 0..layers.len() - 1 {
        let src = &layers[i];
        let dst = &layers[i + 1];
        let dst_layer = &net.layers[i];

        for (dst_i, &dst_idx) in dst.nodes.iter().enumerate() {
            let neuron = &dst_layer.neurons[dst_i];
            for (src_i, &src_idx) in src.nodes.iter().enumerate() {
                let w = neuron.weights[src_i];
                let _ = g.add_edge_with_label(src_idx, dst_idx, (), format!("{:.3}", w));
            }
        }
    }

    (g, layers)
}

// ── Colour apply ──────────────────────────────────────────────────

fn apply_visibility(g: &mut NetGraph, layers: &[LayerMeta], solo_layer: Option<usize>) {
    for (li, meta) in layers.iter().enumerate() {
        let color = if !meta.visible {
            HIDDEN
        } else if solo_layer.is_none() || Some(li) == solo_layer {
            match li {
                0 => COLOR_INPUT,
                l if l == layers.len() - 1 => COLOR_OUTPUT,
                _ => COLOR_HIDDEN,
            }
        } else {
            match li {
                0 => DIM_INPUT,
                l if l == layers.len() - 1 => DIM_OUTPUT,
                _ => DIM_HIDDEN,
            }
        };
        for &n in &meta.nodes {
            if let Some(node) = g.g_mut().node_weight_mut(n) {
                node.set_color(color);
            }
        }
    }
}

// ── Public API ────────────────────────────────────────────────────

/// Open an interactive window visualizing the network's architecture and weights.
///
/// A sidebar on the left lets you toggle layer visibility and "solo" one layer
/// (dimming all others). Nodes are laid out horizontally: input → layers → output.
/// You can zoom and pan the graph freely.
///
/// Blocks until the window is closed.
pub fn visualize(net: &NeuralNetwork) -> Result<(), eframe::Error> {
    let (g, layers) = build_graph_and_meta(net);

    eframe::run_native(
        "Neuron — Network Visualizer",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size(egui::vec2(1280.0, 720.0)),
            ..Default::default()
        },
        Box::new(|_cc| {
            Ok(Box::new(VisualizerApp {
                g,
                layers,
                solo_layer: None,
            }))
        }),
    )
}

// ── App ───────────────────────────────────────────────────────────

struct VisualizerApp {
    g: NetGraph,
    layers: Vec<LayerMeta>,
    solo_layer: Option<usize>,
}

impl eframe::App for VisualizerApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // ── Sidebar ────────────────────────────────────────────────
        let mut changed = false;
        egui::Panel::left("layer_control")
            .resizable(true)
            .default_size(220.0)
            .show_inside(ui, |ui| {
                ui.heading("Layers");
                ui.separator();
                egui::ScrollArea::vertical().show(ui, |ui| {
                    for i in 0..self.layers.len() {
                        let is_solo = self.solo_layer == Some(i);
                        ui.horizontal(|ui| {
                            let mut vis = self.layers[i].visible;
                            if ui.checkbox(&mut vis, &self.layers[i].name).changed() {
                                self.layers[i].visible = vis;
                                changed = true;
                            }
                            if ui
                                .selectable_label(is_solo, if is_solo { "⊕" } else { "⊙" })
                                .on_hover_text("Isolate this layer")
                                .clicked()
                            {
                                self.solo_layer = if is_solo { None } else { Some(i) };
                                changed = true;
                            }
                        });
                    }
                });
                ui.separator();
                if ui.button("Reset all").clicked() {
                    for meta in &mut self.layers {
                        meta.visible = true;
                    }
                    self.solo_layer = None;
                    changed = true;
                }
            });

        if changed {
            apply_visibility(&mut self.g, &self.layers, self.solo_layer);
        }

        // ── Graph (no-op layout preserves manual positions) ─────────
        egui::CentralPanel::default().show_inside(ui, |ui| {
            let mut view = GraphViewWidget::new(&mut self.g)
                .with_navigations(
                    &SettingsNavigation::new()
                        .with_zoom_and_pan_enabled(true)
                        .with_fit_to_screen_enabled(true),
                )
                .with_styles(&SettingsStyle::new().with_labels_always(true));
            ui.add(&mut view);
        });
    }
}
