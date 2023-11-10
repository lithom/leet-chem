package tech.molecules.leet.chem.sar.analysis;

import org.apache.commons.lang3.tuple.Pair;

import javax.swing.table.AbstractTableModel;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class SinglePartAnalysisTableModel extends AbstractTableModel {

    private NumericDecompositionProvider decomposition;

    public SinglePartAnalysisTableModel(NumericDecompositionProvider decomposition, List<String> partLabels) {
        this.decomposition = decomposition;
        this.setPart(partLabels);
    }

    private List<PartData> data = new ArrayList<>();
    private String[] columnNames = {"Variant", "N", "Average", "Median", "Min", "Max", "Standard Deviation"};

    //public SinglePartAnalysisTableModel() {
    //    data = new ArrayList<>();
    //}

    public void setPart(List<String> partLabels) {
        // TODO: Populate the 'data' list based on the given part.
        // This would involve computing the statistics for each variant of the part.
        List<PartData> newData = new ArrayList<>();
        Map<Part,List<Pair<String,Part>>> variants = this.decomposition.getAllVariantsForPart2(partLabels);
        for(Part pi : variants.keySet()) {
            DescriptiveStats.Stats ds = decomposition.getStatsForStructures(variants.get(pi));
            String combinedPart = pi.getVariants().stream().map(xi -> xi.getLeft()+"->"+xi.getRight()).collect(Collectors.joining(","));
            DescriptiveStats.Stats stats_i = ds;
            newData.add(new PartData(combinedPart,stats_i));
        }
        this.data = newData;
        fireTableDataChanged();  // Notify the JTable that data has changed
    }

    @Override
    public int getRowCount() {
        return data.size();
    }

    @Override
    public int getColumnCount() {

        return columnNames.length;
    }

    @Override
    public Object getValueAt(int rowIndex, int columnIndex) {
        PartData partData = data.get(rowIndex);
        switch (columnIndex) {
            case 0: return partData.variant;
            case 1: return partData.stats.n;
            case 2: return partData.stats.average;
            case 3: return partData.stats.median;
            case 4: return partData.stats.min;
            case 5: return partData.stats.max;
            case 6: return partData.stats.standardDeviation;
            default: return null;
        }
    }

    @Override
    public String getColumnName(int col) {
        return columnNames[col];
    }

    private class PartData {
        String variant;
        DescriptiveStats.Stats stats;
        public PartData(String variant, DescriptiveStats.Stats stats) {
            this.variant = variant;
            this.stats = stats;
        }
    }
}

