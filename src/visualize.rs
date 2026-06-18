use crate::NeuralNetwork;
use eframe::egui::Color32;
use egui_graphs::{
    DefaultEdgeShape, DefaultGraphView, DefaultNodeShape, Graph as EguiGraph, SettingsNavigation,
    SettingsStyle,
};

type NetGraph = EguiGraph<
    (),
    (),
    petgraph::Directed,
    petgraph::stable_graph::DefaultIx,
    DefaultNodeShape,
    DefaultEdgeShape,
>;

fn build_graph(net: &NeuralNetwork) -> NetGraph {
    let mut g = NetGraph::new(Default::default());

    let x_input = 0.0_f32;
    let x_spacing = 300.0_f32;
    let y_spacing = 80.0_f32;

    // Track nodes per layer for layout
    let mut layer_positions: Vec<Vec<petgraph::stable_graph::NodeIndex>> = Vec::new();

    // Input layer: create one node per input feature
    let n_inputs = net.layers[0].neurons[0].input.len();
    let mut input_nodes = Vec::with_capacity(n_inputs);
    let y_start_in = -(n_inputs as f32 - 1.0) * y_spacing / 2.0;
    for i in 0..n_inputs {
        let y = y_start_in + i as f32 * y_spacing;
        let idx = g.add_node_with_label_and_location(
            (),
            format!("x{i}"),
            eframe::egui::Pos2::new(x_input, y),
        );
        g.g_mut()
            .node_weight_mut(idx)
            .unwrap()
            .set_color(Color32::from_rgb(70, 130, 220));
        input_nodes.push(idx);
    }
    layer_positions.push(input_nodes);

    // Hidden and output layers
    for (layer_idx, layer) in net.layers.iter().enumerate() {
        let x = x_input + (layer_idx as f32 + 1.0) * x_spacing;
        let n_neurons = layer.neurons.len();
        let mut layer_nodes = Vec::with_capacity(n_neurons);
        let y_start = -(n_neurons as f32 - 1.0) * y_spacing / 2.0;

        for (neuron_idx, neuron) in layer.neurons.iter().enumerate() {
            let y = y_start + neuron_idx as f32 * y_spacing;

            // Determine node label and color by layer
            let (label, color) = if layer_idx == net.layers.len() - 1 {
                // Output layer
                (
                    format!("out  b={:.3}", neuron.bias),
                    Color32::from_rgb(70, 200, 120),
                )
            } else {
                // Hidden layer
                (
                    format!("h{}  b={:.3}", neuron_idx, neuron.bias),
                    Color32::from_rgb(220, 180, 70),
                )
            };

            let idx = g.add_node_with_label_and_location(
                (),
                label,
                eframe::egui::Pos2::new(x, y),
            );
            g.g_mut().node_weight_mut(idx).unwrap().set_color(color);
            layer_nodes.push(idx);
        }
        layer_positions.push(layer_nodes);
    }

    // Connect each layer to the next
    for src_layer_idx in 0..layer_positions.len() - 1 {
        let src_nodes = &layer_positions[src_layer_idx];
        let dst_nodes = &layer_positions[src_layer_idx + 1];
        let dst_layer = &net.layers[src_layer_idx]; // weights live on the receiving layer

        for (dst_i, &dst_idx) in dst_nodes.iter().enumerate() {
            let neuron = &dst_layer.neurons[dst_i];
            for (src_i, &src_idx) in src_nodes.iter().enumerate() {
                let w = neuron.weights[src_i];
                let label = format!("{:.3}", w);
                let _ = g.add_edge_with_label(src_idx, dst_idx, (), label);
            }
        }
    }

    g
}

/// Open an interactive window visualizing the network's architecture and weights.
///
/// The graph shows each neuron as a node (coloured by layer) and each weight as
/// a labelled directed edge. You can zoom, pan, and drag nodes.
///
/// Blocks until the window is closed.
pub fn visualize(net: &NeuralNetwork) -> Result<(), eframe::Error> {
    let g = build_graph(net);

    eframe::run_native(
        "Neuron — Network Visualizer",
        eframe::NativeOptions::default(),
        Box::new(|_cc| Ok(Box::new(VisualizerApp { g }))),
    )
}

struct VisualizerApp {
    g: NetGraph,
}

impl eframe::App for VisualizerApp {
    fn ui(&mut self, ui: &mut eframe::egui::Ui, _frame: &mut eframe::Frame) {
        eframe::egui::CentralPanel::default().show_inside(ui, |ui| {
            let mut view = DefaultGraphView::new(&mut self.g)
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
