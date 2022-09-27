package tech.molecules.leet.similarity.gui;

import org.apache.commons.lang3.tuple.Pair;
import org.jfree.chart.renderer.PaintScale;
import tech.molecules.leet.clustering.ClusterAppModel;
import tech.molecules.leet.table.NClassification;
import tech.molecules.leet.table.NexusTableModel;
import tech.molecules.leet.table.chart.JFreeChartScatterPlot;
import tech.molecules.leet.table.chart.JFreeChartScatterPlot2;
import tech.molecules.leet.table.chart.JScatterPlotConfigurationMenuBar;
import tech.molecules.leet.table.chart.ScatterPlotModel;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.util.List;
import java.util.*;
import java.util.stream.Collectors;

public class JUmapView extends JPanel { //implements JLinkGlassPane.LinkProvider {

    private UmapViewModel model;


    // View configuration:
    private boolean showClusteringAnnotations = true;

    private JPanel jp_top;
    private JPanel jp_plots;
    private JMenuBar jmb_main;

    public JUmapView(UmapViewModel model) {
        this.model = model;
        this.reinit();
        //this.initContextMenu();
    }

    private List<JFreeChartScatterPlot2> plots = new ArrayList<>();

    private void reinit() {
        this.removeAll();


        List<ScatterPlotModel> plots_n = this.model.createPlots();

        this.jp_top = new JPanel();
        this.jp_plots = new JPanel();
        this.setLayout(new BorderLayout());

        this.jmb_main = new JMenuBar();

        this.add(jp_plots,BorderLayout.CENTER);
        this.add(jp_top,BorderLayout.NORTH);

        this.jp_top.setLayout(new BorderLayout());
        this.jp_top.add(this.jmb_main,BorderLayout.CENTER);


        jp_plots.setLayout(new GridLayout(2, 2));
        for(int zi=0;zi<4;zi++) {
            JFreeChartScatterPlot2 jfcsp = new JFreeChartScatterPlot2(plots_n.get(zi),false);
            this.plots.add(jfcsp);
            this.jp_plots.add(this.plots.get(zi));
        }
        JScatterPlotConfigurationMenuBar.initMenuBar(this.jmb_main,model.getNexusTableModel(),plots.stream().map(pi -> pi.getModel()).collect(Collectors.toList()));

        JFreeChartScatterPlot.ScatterPlotListener listener = new JFreeChartScatterPlot.ScatterPlotListener() {
            @Override
            public void highlightingChanged(NexusTableModel.NexusHighlightingChangedEvent e) {
                contactPositions.clear();
                for(JFreeChartScatterPlot2 pi : plots) {
                    // todo: add upper left corner etc. pos
                    pi.getModel().setHighlight(e.getRows(),false);
                    for(String pki : e.getRows()) {
                        double pkip[] = pi.getPositionOfKey(pki);
                        contactPositions.put(pki, Collections.singletonList(pkip));
                    }
                }
                model.getClusterAppModel().setHighlighted( new ArrayList<>( e.getRows() ) );
            }
            @Override
            public void selectionChanged(NexusTableModel.NexusSelectionChangedEvent e) {

            }
        };

        // add model listener to synchronize selection:
        this.model.getClusterAppModel().addClusterAppModelListener(new ClusterAppModel.ClusterAppModelListener() {
            @Override
            public void selectionChanged(NexusTableModel.NexusSelectionChangedEvent e) {
                for(JFreeChartScatterPlot2 pi : plots) {
                    pi.getModel().setSelection(e.getRows());
                    pi.getModel().setHighlightNNearestNeighbors(-1);
                }
            }

            @Override
            public void highlightingChanged(NexusTableModel.NexusHighlightingChangedEvent e) {
                for(JFreeChartScatterPlot2 pi : plots) {
                    pi.getModel().setHighlight(new HashSet<>(e.getRows()),false);
                }
            }

            @Override
            public void clustersChanged() {
                if(showClusteringAnnotations) {
                    Action colorByClustering = new AbstractAction() {
                        @Override
                        public void actionPerformed(ActionEvent e) {
                            List<Pair<Color, List<String>>> coloring = model.getClusterAppModel().getClusters().stream().map(si -> Pair.of(si.getColor(),(List<String>)new ArrayList(si.getMembers()))).collect(Collectors.toList());
                            for (JFreeChartScatterPlot2 csp : plots) {
                                csp.getModel().new SetClusteringAnnotationsAction("", coloring).actionPerformed(e);
                            }
                        }
                    };
                    colorByClustering.actionPerformed(null);
                }
            }
        });

        for(JFreeChartScatterPlot2 pi : this.plots) {
            pi.getModel().addScatterPlotListener(listener);
            pi.getModel().setHighlightNNearestNeighbors(5);
            pi.getModel().setWithoutAxisAndLegend();
        }

        initContextMenu();
    }

    public void initContextMenu() {
        Action colorAnnotationsByClustering = new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent e) {
                List<Pair<Color, List<String>>> coloring = model.getClusterAppModel().getClusters().stream().map(si -> Pair.of(si.getColor(), (List<String>) (new ArrayList<>(si.getMembers())))).collect(Collectors.toList());
                for (JFreeChartScatterPlot2 csp : plots) {
                    csp.getModel().new SetClusteringAnnotationsAction("", coloring).actionPerformed(e);
                }
            }
        };
        colorAnnotationsByClustering.putValue(Action.NAME, "Set annotations by clustering");

        Action colorByClustering = new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent e) {
                //Map<String,Color> colors = new HashMap<>();
                Map<ClusterAppModel.Cluster,Integer> clusterintmap = new HashMap<>();
                Map<ClusterAppModel.Cluster,Color>  clustercolmap = new HashMap<>();
                Map<String,Integer> clustervalues = new HashMap<>();
                for(int zi=0;zi<model.getClusterAppModel().getClusters().size();zi++) {
                    clusterintmap.put(model.getClusterAppModel().getClusters().get(zi),zi);
                    clustercolmap.put(model.getClusterAppModel().getClusters().get(zi),model.getClusterAppModel().getClusters().get(zi).getColor());
                }
                for(String si : model.getNexusTableModel().getAllRows()) {
                    List<NClassification.NClass> ci = model.getClusterAppModel().getClassification().getClassesForRow(si);
                    if(ci.size()>0) {
                        clustervalues.put( si , clusterintmap.get( ci.get(0) ) );
                    }
                    else {
                        clustervalues.put( si , -1);
                    }
                }
                PaintScale psc = new PaintScale() {
                    @Override
                    public double getLowerBound() {
                        return -1.5;
                    }
                    @Override
                    public double getUpperBound() {
                        return model.getClusterAppModel().getClusters().size()+1;
                    }
                    @Override
                    public Paint getPaint(double v) {
                        if(v>=0) {return model.getClusterAppModel().getClusters().get( (int)v ).getColor(); }
                        return Color.gray;
                    }
                };
                for (JFreeChartScatterPlot2 csp : plots) {
                    csp.getModel().setColorExpclicit( psc , clustervalues );
                }
            }
        };
        colorByClustering.putValue(Action.NAME, "Set color by clustering");

        JPopupMenu jpop = new JPopupMenu();
        //jpop.add(new JMenuItem("Select"));

        jpop.add(new SelectionAction());
        jpop.add(new UnselectAllAction());
        jpop.add(new ActivateNeighborhoodHighlightingAction());
        jpop.add(colorByClustering);

        for(JFreeChartScatterPlot2 csp : plots) {
            csp.getChartPanel().setPopupMenu(jpop);
        }
    }

    /*
     public void initContextMenu() {

     List<Action> coloring_actions = new ArrayList<>();
     FixedNumericalDataColumn fndc_none = new FixedNumericalDataColumn("none",new HashMap<>());
     List<Pair<NColumn, NDataset>> data_cols = new ArrayList<>( model.getNexusTableModel().getNexusColumns());
     data_cols.add(Pair.of(fndc_none,model.getStructureProvider()));
     for(Pair<NColumn, NDataset> cpi : data_cols) {
     if( cpi.getLeft() instanceof FixedNumericalDataColumn )  {
     Action gai = new AbstractAction() {
    @Override
    public void actionPerformed(ActionEvent e) {
    Map<String,Double> numdata = ((FixedNumericalDataColumn)(cpi.getLeft())).getDataset();
    for(JFreeChartScatterPlot csp : plots) {
    csp.new SetColoringAction( cpi.getLeft().getName() , numdata).actionPerformed(e);
    }
    //                        p1.new SetColoringAction( cpi.getLeft().getName() , numdata).actionPerformed(e);
    //                        p2.new SetColoringAction( cpi.getLeft().getName() , numdata).actionPerformed(e);
    //                        p3.new SetColoringAction( cpi.getLeft().getName() , numdata).actionPerformed(e);
    //                        p4.new SetColoringAction( cpi.getLeft().getName() , numdata).actionPerformed(e);
    }
    };
     gai.putValue(Action.NAME,cpi.getLeft().getName());
     coloring_actions.add( gai );
     }
     }

     //if(false) {
     Action colorByClustering = new AbstractAction() {
    @Override
    public void actionPerformed(ActionEvent e) {
    List<Pair<Color, List<String>>> coloring = model.getClusterAppModel().getClusters().stream().map(si -> Pair.of(si.getColor(), si.getStructures())).collect(Collectors.toList());
    for (JFreeChartScatterPlot csp : plots) {
    csp.new SetClusteringAnnotationsAction("", coloring).actionPerformed(e);
    }
    }
    };
     colorByClustering.putValue(Action.NAME, "By clustering");
     //}

     List<Pair<String, Colormap>> cms = new ArrayList<>();
     cms.add(Pair.of("CubeYF", Colormaps.get("CubeYF")));
     cms.add(Pair.of("HSV", Colormaps.get("HSV")));

     List<Action> colormap_actions = new ArrayList<>();
     for(Pair<String,Colormap> cmi : cms) {
     Action gai = new AbstractAction() {
    @Override
    public void actionPerformed(ActionEvent e) {
    for(JFreeChartScatterPlot csp : plots) {
    csp.new SetColormapAction( cmi.getLeft() , cmi.getRight()).actionPerformed(e);
    }
    //                    p1.new SetColormapAction( cmi.getLeft() , cmi.getRight()).actionPerformed(e);
    //                    p2.new SetColormapAction( cmi.getLeft() , cmi.getRight()).actionPerformed(e);
    //                    p3.new SetColormapAction( cmi.getLeft() , cmi.getRight()).actionPerformed(e);
    //                    p4.new SetColormapAction( cmi.getLeft() , cmi.getRight()).actionPerformed(e);
    }
    };
     gai.putValue(Action.NAME,cmi.getLeft());
     colormap_actions.add(gai);
     }


     // Set popup menu
     JPopupMenu jpop = new JPopupMenu();

     //jpop.add(new JMenuItem("Select"));
     jpop.add(new SelectionAction());
     jpop.add(new UnselectAllAction());
     JMenu jm_coloring = new JMenu("Set coloring");
     JMenu jm_annotations = new JMenu("Set annotation");
     JMenu jm_coloring_value = new JMenu("By value");
     jm_coloring.add(jm_coloring_value);
     for(Action aci : coloring_actions) { jm_coloring_value.add(aci); }

     JMenuItem jm_annotations_byClustering = new JMenuItem(colorByClustering);
     jm_annotations.add(jm_annotations_byClustering);
     JMenu jm_annotations_bySpecificCluster = new JMenu("By specific cluster");

     JMenu jm_colormap = new JMenu("Set colormap");
     for(Action aci : colormap_actions) {jm_colormap.add(aci);}
     //cms.stream().forEach( ci -> jm_colormap.add( new SetColormapAction( ci.getLeft() , ci.getRight()) ) );

     jpop.add(jm_coloring);
     jpop.add(jm_annotations);
     jpop.add(jm_colormap);

     for(JFreeChartScatterPlot csp : this.plots) {
     //csp.initContextMenu(coloring_actions,colormap_actions);
     csp.setContextMenu(jpop);
     }
     //        p1.initContextMenu(coloring_actions,colormap_actions);
     //        p2.initContextMenu(coloring_actions,colormap_actions);
     //        p3.initContextMenu(coloring_actions,colormap_actions);
     //        p4.initContextMenu(coloring_actions,colormap_actions);
     }


     */


    private Map<String, List<double[]>> contactPositions = new HashMap<>();
    //@Override
    public Map<String, List<double[]>> getLinkContacts() {
        return contactPositions;
    }

    //@Override
    public double[] getLeftUpperCorner() {
        return new double[]{this.getLocation().x,this.getLocation().y};
    }

    private class SelectionAction extends AbstractAction {
        public SelectionAction() {
            super("Select");
        }
        @Override
        public void actionPerformed(ActionEvent e) {
            model.getClusterAppModel().setSelection( model.getClusterAppModel().getHighlighted() );
        }
    }

    /*
    @Override
    public void addLinkProviderListener(JLinkGlassPane.LinkProviderListener li) {

    }

    @Override
    public boolean removeLinkProviderListener(JLinkGlassPane.LinkProviderListener li) {
        return false;
    }

    */

    private class UnselectAllAction extends AbstractAction {
        public UnselectAllAction() {
            super("Unselect all");
        }
        @Override
        public void actionPerformed(ActionEvent e) {
            model.getClusterAppModel().setSelection( new ArrayList<>() );
        }
    }

    private class ActivateNeighborhoodHighlightingAction extends AbstractAction {
        public ActivateNeighborhoodHighlightingAction() {
            super("Activate highlighting");
        }
        @Override
        public void actionPerformed(ActionEvent e) {
            plots.forEach( pi -> pi.getModel().setHighlightNNearestNeighbors(5) );
        }
    }


}
