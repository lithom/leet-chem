package tech.molecules.leet.gui.chem.echelon.view;

import tech.molecules.leet.gui.chem.echelon.model.PartAnalysisModel;

import javax.swing.*;
import java.awt.*;
import java.util.List;

public class PartDetailView extends JPanel {

    private PartAnalysisModel model;
    private List<String> partLabels;


    private JTable table;
    private JScrollPane scrollPane;
    private JPanel topArea;

    public PartDetailView(PartAnalysisModel model, List<String> partLabels) {
        this.model = model;
        this.partLabels = partLabels;
        reinit();
    }

    private void reinit() {
        this.removeAll();
        setLayout(new BorderLayout());

        // Initialize the top area
        topArea = new JPanel();
        topArea.setLayout(new FlowLayout(FlowLayout.LEFT));
        add(topArea, BorderLayout.NORTH);

        // Initialize the table
        table = new JTable(this.model.getSinglePartAnalysisTableModel(partLabels));
        scrollPane = new JScrollPane(table);
        add(scrollPane, BorderLayout.CENTER);

    }

}
