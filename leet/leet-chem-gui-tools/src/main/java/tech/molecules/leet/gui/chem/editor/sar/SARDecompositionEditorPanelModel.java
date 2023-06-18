package tech.molecules.leet.gui.chem.editor.sar;

public class SARDecompositionEditorPanelModel {

    private SARDecompositionFragmentListModel listModel;

    public SARDecompositionEditorPanelModel() {
        this.listModel = new SARDecompositionFragmentListModel();
    }

    public SARDecompositionFragmentListModel getListModel() {
        return listModel;
    }
}
