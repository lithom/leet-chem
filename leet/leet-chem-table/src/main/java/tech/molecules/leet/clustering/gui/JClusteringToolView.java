package tech.molecules.leet.clustering.gui;

import tech.molecules.leet.clustering.ClusterAppModel;
import tech.molecules.leet.similarity.gui.JUmapView;
import tech.molecules.leet.similarity.gui.UmapViewModel;
import tech.molecules.leet.table.NexusTableModel;
import tech.molecules.leet.table.gui.JSimpleClusterView;
import tech.molecules.leet.table.gui.JStructureGridPanel;

import javax.swing.*;
import java.awt.*;

public class JClusteringToolView extends JPanel {

    private ClusterAppModel model;

    private UmapViewModel umapModel;
    private JStructureSelectionPanel.StructureSelectionModel selectionModel;

    private JSplitPane jsp_main;
    private JSimpleClusterView jscv;

    private JStructureSelectionPanel jsgp;

    private JUmapView jumap;

    public JClusteringToolView(ClusterAppModel model) {
        this.model = model;
        umapModel = new UmapViewModel(model);
        selectionModel = new JStructureSelectionPanel.StructureSelectionModel(umapModel);

        reinit();
    }

    private void reinit() {
        this.removeAll();
        this.setLayout(new BorderLayout());

        this.jsp_main = new JSplitPane(JSplitPane.VERTICAL_SPLIT);
        this.add(jsp_main,BorderLayout.CENTER);

        this.jscv = new JSimpleClusterView(model.getStructureProvider(),model);
        this.jsp_main.setBottomComponent(this.jscv);

        this.jumap = new JUmapView(this.umapModel);
        JPanel p_top = new JPanel();
        JSplitPane jsp_top = new JSplitPane();
        p_top.setLayout(new BorderLayout());
        p_top.add(jsp_top,BorderLayout.CENTER);
        jsp_top.setLeftComponent(this.jumap);
        //p_top.add(this.jumap,BorderLayout.CENTER);
        this.jsp_main.setTopComponent(p_top);
        this.jsgp = new JStructureSelectionPanel(selectionModel);
        jsp_top.setRightComponent(this.jsgp);
        //p_top.add(this.jsgp,BorderLayout.EAST);



    }



}
