package tech.molecules.leet.similarity.gui;

import com.actelion.research.chem.descriptor.DescriptorHandlerFunctionalGroups;
import com.actelion.research.chem.descriptor.DescriptorHandlerHashedCFp;
import com.actelion.research.chem.descriptor.DescriptorHandlerLongFFP512;
import com.actelion.research.chem.descriptor.DescriptorHandlerSkeletonSpheres;

import org.apache.commons.lang3.tuple.Pair;
import tech.molecules.leet.clustering.ClusterAppModel;
import tech.molecules.leet.similarity.UMapHelper;
import tech.molecules.leet.table.*;
import tech.molecules.leet.table.chart.JFreeChartScatterPlot2;
import tech.molecules.leet.table.chart.ScatterPlotModel;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class UmapViewModel {

    private ClusterAppModel clusterAppModel;
    //private NexusTableModel ntm;
    //private NStructureDataProvider dsp;

    private List<Pair<NColumn, NStructureDataProvider>> similarityColumns = new ArrayList<>();

    public UmapViewModel(ClusterAppModel clusterAppModel) {
        this.clusterAppModel = clusterAppModel;
        //this.initSimilarities(clusterAppModel.getNtm().getAllRows());
        this.initSimilarities(clusterAppModel.getNtm().getVisibleRows());
    }

    public void initSimilarities(List<String> rowids) {
        NStructureDataProvider dsp = this.clusterAppModel.getStructureProvider();
        PairwiseDistanceColumn nc_ffp = PairwiseDistanceColumn.createFromDescriptor(dsp, DescriptorHandlerLongFFP512.getDefaultInstance(), new ArrayList<>(rowids));
        PairwiseDistanceColumn nc_ffg = PairwiseDistanceColumn.createFromDescriptor(dsp, DescriptorHandlerFunctionalGroups.getDefaultInstance(), new ArrayList<>(rowids));
        PairwiseDistanceColumn nc_skl = PairwiseDistanceColumn.createFromDescriptor(dsp, DescriptorHandlerSkeletonSpheres.getDefaultInstance(), new ArrayList<>(rowids));
        PairwiseDistanceColumn nc_flexo = PairwiseDistanceColumn.createFromDescriptor(dsp, DescriptorHandlerHashedCFp.getDefaultInstance(), new ArrayList<>(rowids));//PairwiseDistanceColumn.createFromDescriptor(dsp, DescriptorHandlerFlexophore.getDefaultInstance(),new ArrayList<>(data.keySet()));

        similarityColumns = new ArrayList<>();
        similarityColumns.add(Pair.of(nc_ffp,dsp));
        similarityColumns.add(Pair.of(nc_ffg,dsp));
        similarityColumns.add(Pair.of(nc_skl,dsp));
        similarityColumns.add(Pair.of(nc_flexo,dsp));
    }

    public List<ScatterPlotModel> createPlots() {
        List<PairwiseDistanceColumn> columns = this.getSimilarityColumns();
        NStructureDataProvider dsp = this.clusterAppModel.getStructureProvider();
        NexusTableModel ntm = this.getNexusTableModel();
        UMapHelper.UMapXYChartConfig config_a = new UMapHelper.UMapXYChartConfig(dsp, columns.get(0));
        UMapHelper.UMapXYChartConfig config_b = new UMapHelper.UMapXYChartConfig(dsp, columns.get(1));
        UMapHelper.UMapXYChartConfig config_c = new UMapHelper.UMapXYChartConfig(dsp, columns.get(2));
        UMapHelper.UMapXYChartConfig config_d = new UMapHelper.UMapXYChartConfig(dsp, columns.get(3));
        ScatterPlotModel p1 = UMapHelper.createChart2(ntm, config_a);
        ScatterPlotModel p2 = UMapHelper.createChart2(ntm, config_b);
        ScatterPlotModel p3 = UMapHelper.createChart2(ntm, config_c);
        ScatterPlotModel p4 = UMapHelper.createChart2(ntm, config_d);
        List<ScatterPlotModel> plots = new ArrayList<>();
        plots.add(p1);plots.add(p2);plots.add(p3);plots.add(p4);
        return plots;
    }



    public ClusterAppModel getClusterAppModel() {
        return clusterAppModel;
    }

    public NStructureDataProvider getStructureProvider() {
        return this.clusterAppModel.getStructureProvider();
    }

    public List<PairwiseDistanceColumn> getSimilarityColumns() {
        return this.similarityColumns.stream().map( ci -> (PairwiseDistanceColumn) ci.getLeft() ).collect(Collectors.toList());
    }

    public NexusTableModel getNexusTableModel() {
        return this.clusterAppModel.getNtm();
    }

    public static interface UmapViewListener {
        public void highlightingChanged();
        public void selectionChanged();
    }

    /**
    private List<UmapViewListener> listeners;

    private void fireSelectionChanged

    public void addUmapViewListener(UmapViewListener li) {
        this.listeners.add(li);
    }

    public void removeUmapViewListener(UmapViewListener li) {
        this.listeners.remove(li);
    }
    **/
}
