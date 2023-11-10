package tech.molecules.leet.gui.chem.echelon.model;

import tech.molecules.leet.chem.sar.SimpleSARDecomposition;
import tech.molecules.leet.chem.sar.SimpleSARDecompositionModel;
import tech.molecules.leet.chem.sar.SimpleSARSeries;
import tech.molecules.leet.chem.sar.analysis.DecompositionProvider;
import tech.molecules.leet.chem.sar.analysis.DecompositionProviderFactory;
import tech.molecules.leet.chem.sar.analysis.DefaultNumericDecompositionProvider;
import tech.molecules.leet.chem.sar.analysis.NumericDecompositionProvider;

import javax.swing.table.AbstractTableModel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

public class EchelonModel {

    private Map<String,Double> numericData;

    private Function<Double,Double> scoreFunction = aDouble -> aDouble;

    /**
     *
     * @param sf used for ranking the numeric values for compounds.
     */
    public void setScoreFunction(Function<Double,Double> sf) {
        this.scoreFunction = sf;
    }

    public void setNumericData(Map<String,Double> nd) {
        this.numericData = nd;
    }

    /**
     *
     * Returns value and score.
     * Returns null in case that there is no value for the given compound!
     *
     * @return {value,score} or Null
     */
    public double[] getValueForCompound(String compound) {
        if(this.numericData.containsKey(compound)) {
            double vi = this.numericData.get(compound);
            double si = this.scoreFunction.apply(vi);
            return new double[]{vi,si};
        }
        return null;
    }

    private List<String> compounds = new ArrayList<>();

    private SimpleSARDecompositionModel decompositionModel = new SimpleSARDecompositionModel(new ArrayList<>());

    private SARTableModel sarTableModel = new SARTableModel();

    private CompoundTableModel compoundTableModel = new CompoundTableModel();

    public EchelonModel() {
        this.reinit();
    }

    private void reinit() {
        decompositionModel = new SimpleSARDecompositionModel(new ArrayList<>());
        decompositionModel.addListener(new SimpleSARDecompositionModel.DecompositionModelListener() {
            @Override
            public void newDecompositions() {
                fireDecompositionsChanged();
            }
        });
    }

    private void fireDecompositionsChanged() {
        this.sarTableModel.fireTableDataChanged();
        this.compoundTableModel.fireTableDataChanged();
    }

    public void setCompounds(List<String> idcodes) {
        this.compounds = new ArrayList<>(idcodes);
        decompositionModel.addCompounds(idcodes);
    }

    public List<String> getCompounds() {
        return this.compounds;
    }

    public SimpleSARDecompositionModel getDecompositionModel() {
        return decompositionModel;
    }

    public SARTableModel getSarTableModel() {
        return sarTableModel;
    }

    public CompoundTableModel getCompoundTableModel() {
        return compoundTableModel;
    }


    public class SARTableModel extends AbstractTableModel {
        @Override
        public int getRowCount() {
            return 0;
        }

        @Override
        public int getColumnCount() {
            return 0;
        }

        @Override
        public Object getValueAt(int rowIndex, int columnIndex) {
            return null;
        }
    }

    public class CompoundTableModel extends AbstractTableModel {
        @Override
        public int getRowCount() {
            return compounds.size();
        }

        @Override
        public int getColumnCount() {
            return 6;
        }

        @Override
        public Object getValueAt(int rowIndex, int columnIndex) {
            if(rowIndex >= compounds.size()) {
                return "error";
            }
            String idc = compounds.get(rowIndex);
            switch(columnIndex) {
                case 0: return idc;
                case 1: return decompositionModel.getSeriesForCompound(idc);
            }
            // in this case we return part of the decomposition..
            SimpleSARSeries series = decompositionModel.getSeriesForCompound(idc);
            if(series != null) {
                List<String> labels_i = series.getLabels().stream().sorted().collect(Collectors.toList());
                SimpleSARDecomposition.SimpleSARResult ri = decompositionModel.getDecompositionForCompound(idc);
                int idx_label = columnIndex - 2;
                if (ri != null) {
                    if (idx_label < labels_i.size()) {
                        return ri.get(labels_i.get(idx_label)).matchedFrag.getIDCode();
                    }
                }
            }
            return "";
        }
    }


    /**
     *
     * @return NumericDecompositionProvider for all the structures that have a measurement based
     *         on the current numericData. Please note that we use as numeric value the SCORE of
     *         the compounds.
     */
    public NumericDecompositionProvider createNumericDecompositionProvider() {
        DecompositionProviderFactory factory = new DecompositionProviderFactory();
        List<SimpleSARDecomposition.SimpleSARResult> data = new ArrayList<>();

        Map<String,Double> cmpToScore = new HashMap<>();
        for(String si : this.compounds) {
            if(this.numericData.containsKey(si)) {
                data.add( this.getDecompositionModel().getDecompositionForCompound(si) );
                cmpToScore.put(si, getValueForCompound(si)[1]);
            }
        }
        DecompositionProvider dp = factory.createDecompositionProviderFromSimpleSARResult(data);
        NumericDecompositionProvider ndp = new DefaultNumericDecompositionProvider(cmpToScore,dp);
        return ndp;
    }

    public PartAnalysisModel getPartAnalysisModel() {
        return new PartAnalysisModel(createNumericDecompositionProvider());
    }

    public void recompute() {

    }


}
