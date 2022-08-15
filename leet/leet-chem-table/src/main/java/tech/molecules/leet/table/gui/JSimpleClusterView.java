package tech.molecules.leet.table.gui;

import tech.molecules.leet.table.NClassification;
import tech.molecules.leet.table.NDataProvider;
import tech.molecules.leet.table.NexusTableModel;

import javax.swing.*;
import javax.swing.event.ListSelectionEvent;
import javax.swing.event.ListSelectionListener;
import javax.swing.table.TableCellRenderer;
import java.awt.*;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class JSimpleClusterView extends JPanel implements NexusTableModel.NexusTableModelListener {

    //private ClusterAppModel model;
    private NDataProvider.StructureDataProvider dataProvider;
    private NClassification model;

    JScrollPane jsp_ClusterList;
    //JList<ClusterAppModel.Cluster> jl_ClusterList;
    JTable jt_ClusterTable;


    JScrollPane jsp_Structures;
    JStructureGridPanel jp_Structures;

    public JSimpleClusterView(NDataProvider.StructureDataProvider dataProvider, NClassification column) {
        this.dataProvider = dataProvider;
        this.model = column;
        this.reinit();
    }

    private void showCluster(NClassification.NClass c) {
        int lines = c.getMembers().size()/3+1;
        List<String> structures = c.getMembers().stream().sorted().map(si -> this.dataProvider.getStructureData(si).structure[0] ).collect(Collectors.toList());
        this.jp_Structures.setData(structures,3,lines,null,null,null);
        this.jp_Structures.repaint();
    }

    private void reinit() {
        this.removeAll();
        this.setLayout(new BorderLayout());

        this.jsp_ClusterList = new JScrollPane();
        //this.jl_ClusterList = new JList<>( this.model.getClusterListModel() );
        //this.jl_ClusterList.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);


        this.jt_ClusterTable = new JTable( new NClassification.ClassificationTableModel(this.model)); //new JTable(model.getClusterTableModel());
        this.jt_ClusterTable.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
        this.jt_ClusterTable.getColumnModel().getColumn(2).setCellRenderer(new ColorRenderer(false));
        this.jt_ClusterTable.getColumnModel().getColumn(0).setPreferredWidth( getPreferredNameColumnWidth() );
        this.jt_ClusterTable.getColumnModel().getColumn(1).setPreferredWidth( getPreferredSmallColumnWidth() );
        this.jt_ClusterTable.getColumnModel().getColumn(1).setMaxWidth( getPreferredSmallColumnWidth() );
        this.jt_ClusterTable.getColumnModel().getColumn(2).setMaxWidth( getPreferredSmallColumnWidth() );


        this.jsp_ClusterList.setViewportView(this.jt_ClusterTable);
        this.add(this.jsp_ClusterList,BorderLayout.WEST);


        this.jsp_Structures = new JScrollPane();
        this.jp_Structures  = new JStructureGridPanel(new ArrayList<>(),4,4);
        this.jsp_Structures.setViewportView(this.jp_Structures);
        this.add(jsp_Structures,BorderLayout.CENTER);


        // add listeners for synchronization with clusterAppModel..
        //this.model.removeClusterAppModelListener(this);
        //this.model.addClusterAppModelListener(this);
        // TODO: add listener to classification..
        this.model.addClassificationListener(new NClassification.ClassificationListener() {
            @Override
            public void classificationChanged() {
                if(jt_ClusterTable.getSelectedRow()>=0) {
                    showCluster( model.getClasses().get( jt_ClusterTable.getSelectedRow()));
                }
                jt_ClusterTable.repaint();
                jp_Structures.repaint();
                //this.jt_ClusterTable.getColumnModel().getColumn(2).setCellRenderer(new ColorRenderer(true));
                revalidate();
            }

            @Override
            public void classChanged(NClassification.NClass ci) {

            }
        });

        // add selection listener to clusterlist to synchronize
        // the visualization of the cluster:
        this.jt_ClusterTable.getSelectionModel().addListSelectionListener(new ListSelectionListener() {
            @Override
            public void valueChanged(ListSelectionEvent e) {
                if(jt_ClusterTable.getSelectedRow()>=0) {
                    showCluster(model.getClasses().get(jt_ClusterTable.getSelectedRow()));
                }
            }
        });
        this.revalidate();
    }

    private int getPreferredNameColumnWidth() {
        //return (new JLabel("abcd_abcd_abcd_abcd_")).getPreferredSize().width;
        return (new JLabel("abcd_abcd_abcd")).getPreferredSize().width;
    }

    private int getPreferredSmallColumnWidth() {
        return (new JLabel("ColorColor")).getPreferredSize().width;
    }

//    @Override
//    public void selectionChanged(NexusSelectionChangedEvent e) {
//    }
//
//    @Override
//    public void highlightingChanged(NexusHighlightingChangedEvent e) {
//    }
//
//    @Override
//    public void clustersChanged() {
//        if(this.jt_ClusterTable.getSelectedRow()>=0) {
//            this.showCluster( model.getClusters().get( jt_ClusterTable.getSelectedRow()));
//        }
//        this.jt_ClusterTable.repaint();
//        this.jp_Structures.repaint();
//        //this.jt_ClusterTable.getColumnModel().getColumn(2).setCellRenderer(new ColorRenderer(true));
//        this.revalidate();
//    }


    public class ColorRenderer extends JLabel
            implements TableCellRenderer {

        boolean isBordered = true;
        public ColorRenderer(boolean isBordered) {
            this.isBordered = isBordered;
            setOpaque(true); //MUST do this for background to show up.
        }

        public Component getTableCellRendererComponent(
                JTable table, Object color,
                boolean isSelected, boolean hasFocus,
                int row, int column) {
            Color newColor = (Color)color;
            setBackground(newColor);
            return this;
        }
    }

    @Override
    public void nexusTableStructureChanged() {
        reinit();
    }
}
