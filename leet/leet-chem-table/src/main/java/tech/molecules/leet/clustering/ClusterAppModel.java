package tech.molecules.leet.clustering;

import org.apache.commons.lang3.tuple.Pair;
import tech.molecules.leet.table.*;

import javax.swing.*;
import javax.swing.table.AbstractTableModel;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.util.List;
import java.util.*;
import java.util.function.Supplier;
import java.util.stream.Collectors;

public class ClusterAppModel implements NDataProvider.StructureDataProvider, NClassification {

    private NStructureDataProvider dsp;

    @Override
    public NDataProvider.StructureWithID getStructureData(String rowid) {
        return dsp.getStructureData(rowid);
    }

    //@Override
    public List<NClassification.NClass> getClasses() {
        return new ArrayList<>(this.clusters);
    }

    //@Override
    public void addClass(NClass ci) {
        try {
            createCluster(ci.getName(),ci.getColor(),new ArrayList<>(ci.getMembers()));
        } catch (ClusterHandlingException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void removeClass(NClass ci) {
        // not yet supported..
        throw new RuntimeException("Not yet supported");
    }

    private List<ClassificationListener> classificationListeners = new ArrayList<>();

    @Override
    public void addClassificationListener(ClassificationListener li) {
        this.classificationListeners.add(li);
    }

    @Override
    public void removeClassificationListener(ClassificationListener li) {
        this.classificationListeners.remove(li);
    }

    @Override
    public List<NClassification.NClass> getClassesForRow(String id) {
        List<NClassification.NClass> clusters = new ArrayList<>();
        for(NClassification.NClass ic : clusters) {
            if(ic.getMembers().contains(id)) {
                clusters.add(ic);
            }
        }
        return clusters;
    }

    private NexusTableModel ntm;
    //private NexusTableModel ntm2;

    private List<String> highlighted = new ArrayList<>();
    private List<String> selection = new ArrayList<>();
    private List<Cluster> clusters = new ArrayList<>();

    public ClusterAppModel(NexusTableModel ntm, NStructureDataProvider dsp) {
        this.init(ntm);
        this.dsp = dsp;
    }

    public void init(NexusTableModel ntm) {
        this.ntm = ntm;
        //this.ntm2 = ntm2;
    }

    public void setSelection(List<String> s) {
        this.selection.clear();
        this.selection.addAll(s);
        this.fireSelectionChanged();
    }

    public void setHighlighted(List<String> s) {
        this.highlighted.clear();
        this.highlighted.addAll(s);
        this.fireHighlightingChanged();
    }

    public List<String> getHighlighted() {
        return this.highlighted;
    }

    public NStructureDataProvider getStructureProvider() {
        return this.dsp;
    }

    public ClassificationColumn getClassificationColumn() {
        ClassificationColumn cc = new ClassificationColumn(this);
        cc.startAsyncInitialization(ntm,dsp);
        cc.addCellPopupAction(new CreateClusterFromCellAction("Create cluster",cc,""));
        return cc;
    }

    public NexusTableModel getNtm() {
        return this.ntm;
    }

//    public NexusTableModel getNtm2() {
//        return this.ntm2;
//    }

    public static final String CLUSTER_FLAG_DEFINING = "defining";

    public List<Cluster> getClusters() {
        return Collections.unmodifiableList( this.clusters );
    }

    private static Color getRandomColor() {
        Random r = new Random();
        Color col = Color.getHSBColor(r.nextFloat(),0.85f,0.95f);
        return col;
    }

    public List<String> getSelection() {
        return Collections.unmodifiableList(this.selection);
    }

    public class Cluster implements NClassification.NClass {
        private String name;
        private String description;
        private HashSet<String> ids;
        private Map<String,Set<String>> flags;
        private Color color;

        public Cluster(String name, String description, Collection<String> structures) {
            this(name,description,getRandomColor(),structures);
        }

        @Override
        public boolean isMember(String ki) {
            return this.ids.contains(ki);
        }

        @Override
        public Set<String> getMembers() {
            return Collections.unmodifiableSet( this.ids );
        }

        public Cluster(String name, String description, Color col, Collection<String> ids) {
            this.setName(name);
            this.setDescription(description);
            this.ids = new HashSet<>(ids);
            Random r = new Random();
            this.color = col;//
        }

        //@Override
        public void addMembers(Collection<String> s) {
            this.ids.addAll(s);
            fireClustersChanged();
        }

        // huh what were these flags? :) Ahhh.. maybe things like "defining" etc.?
//        @Override
//        public void addFlagToStructure(String structure, String flag) throws ClusterHandlingException {
//            if(!this.structures.contains(structure)) {
//                throw new ClusterHandlingException("Structure not in cluster");
//            }
//            if(!this.flags.containsKey(structure)) { this.flags.put(structure,new HashSet<>()); }
//            this.flags.get(structure).add(flag);
//            fireClustersChanged();
//        }

        @Override
        public String getName() {
            return name;
        }

        //@Override
        public void setName(String name) {
            this.name = name;
        }

        @Override
        public String getDescription() {
            return description;
        }

        //@Override
        public void setDescription(String description) {
            this.description = description;
        }

        @Override
        public Color getColor() {
            return this.color;
        }

        /**
         * @TODO Somehow add sorting capabilities in here, e.g.
         * sort to have defining structures first etc..
         *
         * @return
         */
        //@Override
        public List<String[]> getStructures() {
            return   new ArrayList<>( this.ids.stream().map( si -> dsp.getStructureData(si).structure ).collect(Collectors.toList()) );
        }
        //@Override
        public void setColor(Color c_new) {
            this.color = c_new;
        }
    }

    public Cluster createCluster(String name, Color color, List<String> ids) throws ClusterHandlingException {
        Cluster ci = new Cluster(name,"",color,ids);
        //check if cluster with name already exists:
        if( this.clusters.stream().anyMatch( cx -> cx.getName().equals(name)) ) {
            throw new ClusterHandlingException("Cluster with name already exists");
        }
        this.clusters.add(ci);
        this.fireClustersChanged();
        return ci;
    }

    public Cluster getClusterByName(String c) {
        for(Cluster ci : this.clusters) { if(ci.getName().equals(c)){return ci;} }
        return null;
    }

//    public Map<String,List<Color>> getClusterColoring() {
//        Map<String,List<Color>> ccm = new HashMap<>();
//        for(Cluster ci : this.clusters) {
//            Color cci = ci.getColor();
//            for(String[] sci : ci.getStructures()) {
//                if(!ccm.containsKey(sci)) {ccm.put(sci,new ArrayList<>());}
//                ccm.get(sci).add(cci);
//            }
//        }
//        return ccm;
//    }


    private List<ClusterAppModelListener> listeners = new ArrayList<>();

    public void addClusterAppModelListener(ClusterAppModelListener li) {
        this.listeners.add(li);
    }
    public void removeClusterAppModelListener(ClusterAppModelListener li) {
        this.listeners.remove(li);
    }

    private void fireClustersChanged() {
        for(ClusterAppModelListener li : this.listeners) {
            li.clustersChanged();
        }
        for(NClassification.ClassificationListener li : this.classificationListeners) {
            li.classificationChanged();
        }
        this.clusterListModel.fireClustersChanged();
        this.clusterTableModel.fireClustersChanged();
    }

    private void fireSelectionChanged() {
        for(ClusterAppModelListener li : this.listeners) {
            li.selectionChanged(new NexusTableModel.NexusSelectionChangedEvent(this,new HashSet<>(this.selection)));
        }
    }

    private void fireHighlightingChanged() {
        for(ClusterAppModelListener li : this.listeners) {
            li.highlightingChanged(new NexusTableModel.NexusHighlightingChangedEvent(this,new HashSet<>(this.highlighted)));
        }
    }

    public static interface ClusterAppModelListener {
        public void selectionChanged(NexusTableModel.NexusSelectionChangedEvent e);
        public void highlightingChanged(NexusTableModel.NexusHighlightingChangedEvent e);
        public void clustersChanged();
    }


    public class ClusterListModel extends AbstractListModel<Cluster> {
        @Override
        public int getSize() {
            return clusters.size();
        }

        @Override
        public Cluster getElementAt(int index) {
            return clusters.get(index);
        }

        public void fireClustersChanged() {
            this.fireContentsChanged(this,0,clusters.size());
        }
    }

    public class ClusterTableModel extends AbstractTableModel {

        @Override
        public String getColumnName(int column) {
            switch(column) {
                case 0: return "Name";
                case 1: return "Size";
                case 2: return "Color";
            }
            return "<ERROR>";
        }

        @Override
        public int getRowCount() {
            return clusters.size();
        }

        @Override
        public int getColumnCount() {
            return 3;
        }

        @Override
        public Object getValueAt(int rowIndex, int columnIndex) {
            if(rowIndex>=clusters.size()){return null;}
            Cluster ci = clusters.get(rowIndex);
            switch(columnIndex){
                case 0: return ci.getName();
                case 1: return ci.getStructures().size();
                case 2: return ci.getColor();
            }
            return null;
        }

        public void fireClustersChanged() {
            //this.fireTableStructureChanged();
            this.fireTableDataChanged();
        }
    }

    private ClusterListModel  clusterListModel   = new ClusterListModel();
    private ClusterTableModel clusterTableModel  = new ClusterTableModel();

    public ClusterListModel getClusterListModel() {
        return clusterListModel;
    }

    public ClusterTableModel getClusterTableModel() {
        return clusterTableModel;
    }



    public static class ClusterHandlingException extends Exception {
        public ClusterHandlingException(String msg) {
            super(msg);
        }
    }

    public class CreateClusterFromCellAction extends NColumn.CellSpecificAction {
        public CreateClusterFromCellAction(String name, NColumn column, String rowid) {
            super(name, column, rowid);
        }
        @Override
        public void actionPerformed(ActionEvent e) {
            try {
                createCluster("C"+String.format("%.3d",Math.random()),Color.blue,Collections.singletonList(getRowId()));
            } catch (ClusterHandlingException clusterHandlingException) {
                clusterHandlingException.printStackTrace();
            }
        }
    }


    public class CreateClusterAction extends AbstractAction {
        private Supplier<List<String>> structureSupplier;
        private Supplier<Pair<String,Color>>       clusterSupplier;
        //private Supplier<Color>        colorSupplier;
        //private String cluster;

        //private NClassification classification;

        /**
         *
         * @param name
         * @param structuresupplier supplier for initial structures. Can be null.
         * @param clustersupplier
         */
        public CreateClusterAction(String name, Supplier<Pair<String,Color>> clustersupplier, Supplier<List<String>> structuresupplier) {
            super(name);
            //this.classification = nci;
            this.structureSupplier = structuresupplier;
            this.clusterSupplier   = clustersupplier;
            //this.colorSupplier     = colorsupplier;
        }
        @Override
        public void actionPerformed(ActionEvent e) {
            List<String> si = new ArrayList<>();
            if(this.structureSupplier!=null) {
                si.addAll( structureSupplier.get() );
            }
            // see if a cluster with this name already exists:
            Pair<String,Color> scp = this.clusterSupplier.get();
            String cluster = scp.getLeft();
            Color  col     = scp.getRight();
            Cluster ci = getClusterByName(cluster);
            if(ci != null) {
                // not good.. we already have a cluster of this name..
                // Todo: show warning?
            }
            else {
                // create new cluster:
                try {
                    createCluster(cluster,col,si);
                } catch (ClusterHandlingException clusterHandlingException) {
                    clusterHandlingException.printStackTrace();
                }
            }
        }
    }

    public class AddMembersToClusterAction extends AbstractAction {
        private Supplier<List<String>> structureSupplier;
        private String cluster;
        public AddMembersToClusterAction(String name, Supplier<List<String>> idsupplier, String cluster) {
            super(name);
            this.structureSupplier = idsupplier;
            this.cluster = cluster;
        }
        @Override
        public void actionPerformed(ActionEvent e) {
            List<String> si = structureSupplier.get();
            Cluster ci = getClusterByName(this.cluster);
            if(ci == null) {
                // show warning?
            }
            else {
                ci.addMembers(si);
            }
        }
    }

}
