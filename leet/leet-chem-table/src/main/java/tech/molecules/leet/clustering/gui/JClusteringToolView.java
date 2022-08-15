package tech.molecules.leet.clustering.gui;

import tech.molecules.leet.clustering.ClusterAppModel;
import tech.molecules.leet.similarity.gui.JUmapView;
import tech.molecules.leet.similarity.gui.UmapViewModel;
import tech.molecules.leet.table.gui.JSimpleClusterView;

import javax.swing.*;
import java.awt.*;

public class JClusteringToolView extends JPanel {

    private ClusterAppModel model;

    private UmapViewModel umapModel;

    private JSplitPane jsp_main;
    private JSimpleClusterView jscv;

    private JUmapView jumap;

    public JClusteringToolView(ClusterAppModel model) {
        this.model = model;
        umapModel = new UmapViewModel(model);
        reinit();
    }

    private void reinit() {
        this.removeAll();
        this.setLayout(new BorderLayout());

        this.jsp_main = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT);
        this.add(jsp_main,BorderLayout.CENTER);

        this.jscv = new JSimpleClusterView(model.getStructureProvider(),model);
        this.jsp_main.setBottomComponent(this.jscv);

        this.jumap = new JUmapView(this.umapModel);
        JPanel p_top = new JPanel();
        p_top.setLayout(new BorderLayout());
        p_top.add(this.jumap,BorderLayout.CENTER);
        this.jsp_main.setTopComponent(p_top);
    }



}
