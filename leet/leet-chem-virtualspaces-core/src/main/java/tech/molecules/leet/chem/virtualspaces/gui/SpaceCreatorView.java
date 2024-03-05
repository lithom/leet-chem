package tech.molecules.leet.chem.virtualspaces.gui;

import com.actelion.research.chem.reaction.Reaction;
import com.actelion.research.gui.table.ChemistryCellRenderer;
import com.formdev.flatlaf.FlatLightLaf;
import tech.molecules.leet.chem.virtualspaces.gui.task.AddBuildingBlockFilesTask;
import tech.molecules.leet.chem.virtualspaces.gui.task.AddReactionDirectoryTask;

import javax.swing.*;
import java.awt.*;

public class SpaceCreatorView {
    private SpaceCreatorController controller;
    private SpaceCreatorModel model;

    private JFrame frame;
    private JTable tableA;
    private JTable tableB;
    private JButton addButtonA;
    private JButton removeButtonA;
    private JButton addButtonB;
    private JButton removeButtonB;

    public SpaceCreatorView(SpaceCreatorController controller, SpaceCreatorModel model) {
        this.controller = controller;
        this.model = model;

        initLAF();
        reinitUI();
    }

    private void initLAF() {
        try {
            UIManager.setLookAndFeel( new FlatLightLaf() );
        } catch( Exception ex ) {
            System.err.println( "Failed to initialize LaF" );
        }
    }

    private void reinitUI() {
        frame = new JFrame("Library Builder 0.1");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setLayout(new GridLayout(2, 1));

        tableA = new JTable(model.getBuildingBlockFileTableModel());
        JScrollPane scrollPaneA = new JScrollPane(tableA);
        JPanel panelA = new JPanel(new BorderLayout());
        panelA.add(scrollPaneA, BorderLayout.CENTER);

        addButtonA = new JButton("Add BuildingBlockFile");
        removeButtonA = new JButton("Remove Selected");
        JPanel buttonsPanelA = new JPanel();
        buttonsPanelA.add(addButtonA);
        buttonsPanelA.add(removeButtonA);
        panelA.add(buttonsPanelA, BorderLayout.SOUTH);

        tableB = new JTable(model.getReactionMechanismTableModel());
        JScrollPane scrollPaneB = new JScrollPane(tableB);
        JPanel panelB = new JPanel(new BorderLayout());
        panelB.add(scrollPaneB, BorderLayout.CENTER);

        tableB.setDefaultRenderer(Reaction.class,new ChemistryCellRenderer(new Dimension(80,80)));
        tableB.setRowHeight(80);

        addButtonB = new JButton("Add ReactionMechanism");
        removeButtonB = new JButton("Remove Selected");
        JPanel buttonsPanelB = new JPanel();
        buttonsPanelB.add(addButtonB);
        buttonsPanelB.add(removeButtonB);
        panelB.add(buttonsPanelB, BorderLayout.SOUTH);

        frame.add(panelA);
        frame.add(panelB);

        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);

        // Placeholder for action listeners
        addActions();
    }

    public JFrame getFrame() {
        return this.frame;
    }

    private void addActions() {
        addButtonA.setAction(new AddBuildingBlockFilesTask(this.controller));
        addButtonB.setAction(new AddReactionDirectoryTask(this.controller));
    }

}
