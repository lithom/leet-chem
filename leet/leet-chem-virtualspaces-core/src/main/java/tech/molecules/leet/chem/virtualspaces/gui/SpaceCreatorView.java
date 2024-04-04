package tech.molecules.leet.chem.virtualspaces.gui;

import com.actelion.research.chem.reaction.Reaction;
import com.actelion.research.gui.VerticalFlowLayout;
import com.actelion.research.gui.table.ChemistryCellRenderer;
import com.formdev.flatlaf.FlatLightLaf;
import tech.molecules.leet.chem.virtualspaces.gui.task.AddBuildingBlockFilesTask;
import tech.molecules.leet.chem.virtualspaces.gui.task.AddReactionDirectoryTask;
import tech.molecules.leet.chem.virtualspaces.gui.task.CreateSpaceTask;

import javax.swing.*;
import java.awt.*;

public class SpaceCreatorView {
    private SpaceCreatorController controller;
    private SpaceCreatorModel model;

    private JFrame frame;

    private JPanel panelMain;

    private JPanel panelTop;
    private JButton buttonCreateSpace;

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
        //frame.setLayout(new VerticalFlowLayout());
        frame.getContentPane().setLayout(new BorderLayout());
        try{
            ImageIcon icon = new ImageIcon(getClass().getResource("/library_builder_icon_a.png"));
            this.getFrame().setIconImage(icon.getImage());
        }
        catch(Exception ex) {ex.printStackTrace();}

        panelMain = new JPanel(new VerticalFlowLayout());

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


        panelTop = new JPanel(new FlowLayout(FlowLayout.LEFT));
        buttonCreateSpace = new JButton();
        panelTop.add(buttonCreateSpace);
        //buttonCreateSpace.setEnabled(false);

        panelMain.add(panelTop);
        panelA.revalidate();
        panelB.revalidate();
        panelMain.add(panelA);
        panelMain.add(panelB);
        frame.getContentPane().add(panelMain,BorderLayout.CENTER);

        frame.pack();
        frame.setSize(1200,1024);
        panelMain.revalidate();
        panelMain.repaint();
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
        buttonCreateSpace.setAction(new CreateSpaceTask(this.controller));
    }

}
