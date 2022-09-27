package tech.molecules.leet.clustering.gui;

import tech.molecules.leet.clustering.ClusterAppModel;
import tech.molecules.leet.similarity.gui.UmapViewModel;
import tech.molecules.leet.table.NDataProvider;
import tech.molecules.leet.table.NexusTableModel;
import tech.molecules.leet.table.gui.JStructureGridPanel;

import javax.swing.*;
import java.awt.*;
import java.util.List;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Provides compound list views for selection and mouse over structures
 * of the umap view.
 *
 */
public class JStructureSelectionPanel extends JPanel { //implements JLinkGlassPane.LinkProvider {

    public static class StructureSelectionModel {
        private UmapViewModel umap;
        public StructureSelectionModel(UmapViewModel umap) {
            this.umap = umap;
            initModel();
        }
        private void initModel() {
            this.umap.getClusterAppModel().addClusterAppModelListener(new ClusterAppModel.ClusterAppModelListener() {
//                @Override
//                public void selectionChanged(NexusSelectionChangedEvent e) {
//                    fireGuiChangedEvent(true,false);
//                }
//
//                @Override
//                public void highlightingChanged(NexusHighlightingChangedEvent e) {
//                    fireGuiChangedEvent(false,false);
//                }


                @Override
                public void selectionChanged(NexusTableModel.NexusSelectionChangedEvent e) {
                    fireGuiChangedEvent(false,false);
                }

                @Override
                public void highlightingChanged(NexusTableModel.NexusHighlightingChangedEvent e) {
                    fireGuiChangedEvent(false,true);
                }

                @Override
                public void clustersChanged() {
                    fireGuiChangedEvent(false,false);
                }
            });
        }

        private List<ModelListener> listeners = new ArrayList<>();
        public void addModelListener(ModelListener li) {listeners.add(li);}
        public void removeModelListener(ModelListener li) {listeners.remove(li);}

        private void fireGuiChangedEvent(boolean setSelectionToFront, boolean setMouseOverToFront) {
            for(ModelListener li : listeners) {li.guiChanged(setSelectionToFront,setMouseOverToFront);}
        }
        public interface ModelListener {
            public void guiChanged(boolean setSelectionToFront, boolean setMouseOverToFront);
        }

        public UmapViewModel getUmapViewModel() {return this.umap;}
        public ClusterAppModel getClusterAppModel() {return this.umap.getClusterAppModel();}

    }


    private StructureSelectionModel model;

    private Map<String,double[]> linkPoints = new HashMap<>();


    // View configuration
    private int selectionRows = 2;
    private int selectionCols = 3;
    private int mouseOverRows = 2;
    private int mouseOverCols = 3;




    private JTabbedPane jtp_Main = new JTabbedPane();
    private JStructureGridPanel jp_mouseOver;
    private JStructureGridPanel jp_selection;

    public JStructureSelectionPanel(StructureSelectionModel model) {
        this.model = model;
        init();
    }


    private int jtpIdxMouseOver = -1;
    private int jtpIdxSelection = -1;
    private void init() {
        this.removeAll();
        BorderLayout borderLayout = new BorderLayout();
        this.setLayout(borderLayout);

        this.jtp_Main = new JTabbedPane();
        this.jp_selection = new JStructureGridPanel(new ArrayList<>(),selectionRows,selectionCols);
        this.jp_mouseOver = new JStructureGridPanel(new ArrayList<>(),mouseOverRows,mouseOverCols);
        this.add(this.jtp_Main,BorderLayout.CENTER);
        this.jtp_Main.addTab("Highlighted",this.jp_mouseOver);
        jtpIdxMouseOver = this.jtp_Main.getSelectedIndex();
        this.jtp_Main.addTab("Selection",this.jp_selection);
        jtpIdxSelection = this.jtp_Main.getSelectedIndex();

        this.model.addModelListener(new StructureSelectionModel.ModelListener(){
            @Override
            public void guiChanged(boolean setSelectionToFront, boolean setMouseOverToFront) {
                if(setSelectionToFront) {
                    jtp_Main.setSelectedComponent(jp_selection);
                }
                if(setMouseOverToFront) {
                    jtp_Main.setSelectedComponent(jp_mouseOver);
                }
                jp_selection.repaint();
                jp_mouseOver.repaint();
            }
        });

        this.model.getClusterAppModel().addClusterAppModelListener(new ClusterAppModel.ClusterAppModelListener() {
//            @Override
//            public void selectionChanged(NexusSelectionChangedEvent e) {
//                int lines = e.getRows().size() / 4 + 1;
//                jp_selection.setData( new ArrayList<>( model.getClusterAppModel().getSelection() ),4, lines , model.getClusterAppModel().getClusterColoring(), null , null);
//                model.getClusterAppModel().getNtm2().setVisibleRowsFilter( new HashSet<>(model.getClusterAppModel().getSelection()) ); // ntm2 is only selected
//            }
//
//            @Override
//            public void highlightingChanged(NexusHighlightingChangedEvent e) {
//                //jp_mouseOver.setData( new ArrayList<>( e.getRows() ),2,3,model.getClusterAppModel().getClusterColoring(),null,null);
//                jp_mouseOver.setData( new ArrayList<>( model.getClusterAppModel().getHighlighted() ),2,3,model.getClusterAppModel().getClusterColoring(),null,null);
//            }


            @Override
            public void selectionChanged(NexusTableModel.NexusSelectionChangedEvent e) {
                int lines = e.getRows().size() / 4 + 1;
                List<NDataProvider.StructureWithID> structures_h =(model.getClusterAppModel().getHighlighted().stream().map(xi -> model.getClusterAppModel().getStructureProvider().getStructureData(xi)).collect(Collectors.toList() ));
                jp_selection.setData( structures_h,4, lines , model.getClusterAppModel().getClusterColoring(), null , null);
                // TODO: interesting, maybe reimplement something like this..
                //model.getClusterAppModel().getNtm2().setVisibleRowsFilter( new HashSet<>(model.getClusterAppModel().getSelection()) ); // ntm2 is only selected
            }

            @Override
            public void highlightingChanged(NexusTableModel.NexusHighlightingChangedEvent e) {
                List<NDataProvider.StructureWithID> structures_h =(model.getClusterAppModel().getHighlighted().stream().map(xi -> model.getClusterAppModel().getStructureProvider().getStructureData(xi)).collect(Collectors.toList()));
                jp_mouseOver.setData( structures_h , 2,3,null,null,null );
            }

            @Override
            public void clustersChanged() {
                // create Context Menu for Selection List
                JPopupMenu jpop_selection = createPopupMenuForStructureGridPanel(model.getClusterAppModel(),jp_selection);
                jp_selection.setContextMenu(jpop_selection);
                jp_selection.setData( new ArrayList<>( model.getClusterAppModel().getSelection().stream().map( xi -> model.getClusterAppModel().getStructureData(xi) ).collect(Collectors.toList()) ),2,3,model.getClusterAppModel().getClusterColoring(),null,null);
            }
        });

        JPopupMenu jpop_selection = createPopupMenuForStructureGridPanel(model.getClusterAppModel(), jp_selection);
        jp_selection.setContextMenu(jpop_selection);
    }


    @Override
    public Dimension getMinimumSize() {
        return new Dimension(400,400);
    }

    @Override
    public Dimension getPreferredSize() {
        return new Dimension(600,600);
    }


    public static JPopupMenu createPopupMenuForStructureGridPanel(ClusterAppModel app, JStructureGridPanel jgp_selection) {
        // create Context Menu for Selection List
        JPopupMenu jpop_selection = new JPopupMenu();
        // TODO:
        //ClusterAppModel.CreateClusterAction ccaction = app.new CreateClusterAction("Create cluster", ()-> JOptionPane.showInputDialog("Cluster name") ,()-> jgp_selection.getSelected());
        ClusterAppModel.CreateClusterAction ccaction = app.new CreateClusterAction("Create cluster", ()-> JClusterAndColorConfig.showDialog(jgp_selection.getThisJPanel().getParent(),app.new Cluster("","",new ArrayList<>())) ,()-> jgp_selection.getSelected().stream().map(xi -> xi.molid).collect(Collectors.toList()) );
        JMenu jm_addToCluster = new JMenu("Add to cluster");
        for(ClusterAppModel.Cluster ci : app.getClusters()) {
            ClusterAppModel.AddMembersToClusterAction addToClusterAction = app.new AddMembersToClusterAction(""+ci.getName(), ()->jgp_selection.getSelected().stream().map(xi -> xi.molid).collect(Collectors.toList()),ci.getName());//()->Collections.singletonList(jgp_selection.getStructureMouseOver()), ci.getName());
            jm_addToCluster.add(addToClusterAction);
        }
        jpop_selection.add(ccaction);
        jpop_selection.add(jm_addToCluster);
        //jgp_selection.setContextMenu(jpop_selection);
        return jpop_selection;
    }


    private Map<String, List<double[]>> linkContacts = new HashMap<>();

//    @Override
//    public Map<String, List<double[]>> getLinkContacts() {
//        return linkContacts;
//    }
//
//    @Override
//    public double[] getLeftUpperCorner() {
//        return new double[]{this.getLocation().x,this.getLocation().y};
//    }
//
//    private void fireLinksChanged() {
//        for(JLinkGlassPane.LinkProviderListener li : linkProviderListeners) {
//            li.linksChanged();
//        }
//    }
//
//    private List<JLinkGlassPane.LinkProviderListener> linkProviderListeners = new ArrayList<>();
//
//    @Override
//    public void addLinkProviderListener(JLinkGlassPane.LinkProviderListener li) {
//        this.linkProviderListeners.add(li);
//    }
//
//    @Override
//    public boolean removeLinkProviderListener(JLinkGlassPane.LinkProviderListener li) {
//        return this.linkProviderListeners.remove(li);
//    }
}
