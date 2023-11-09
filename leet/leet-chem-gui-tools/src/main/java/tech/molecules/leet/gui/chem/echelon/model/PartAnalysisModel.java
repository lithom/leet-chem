package tech.molecules.leet.gui.chem.echelon.model;

import tech.molecules.leet.chem.sar.analysis.NumericDecompositionProvider;
import tech.molecules.leet.chem.sar.analysis.SinglePartAnalysisTableModel;

import javax.swing.table.TableModel;
import java.util.List;

public class PartAnalysisModel {

    private NumericDecompositionProvider decomp;

    public PartAnalysisModel(NumericDecompositionProvider decomp) {
        this.decomp = decomp;
    }

    public TableModel getSinglePartAnalysisTableModel(List<String> labels) {
        SinglePartAnalysisTableModel spatm = new SinglePartAnalysisTableModel(this.decomp, labels);
        //spatm.setPart(labels);
        return spatm;
    }


}
