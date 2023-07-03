package tech.molecules.leet.gui.chem.editor.sar;

import com.actelion.research.chem.Molecule;
import com.actelion.research.chem.StereoMolecule;
import tech.molecules.leet.chem.sar.SimpleMultiSynthonStructure;

import java.util.ArrayList;
import java.util.List;

public class MultiSynthonStructureEditorModel {

    private SimpleMultiSynthonStructure multiStructure;
    private StereoMolecule synthonEditedVersion;
    private StereoMolecule synthonOriginalReference;

    public MultiSynthonStructureEditorModel(SimpleMultiSynthonStructure multiStructure, StereoMolecule editedSynthon) {
        this.multiStructure = multiStructure;

        this.synthonEditedVersion = new StereoMolecule();
        this.synthonOriginalReference = editedSynthon;

        this.synthonOriginalReference.copyMolecule(this.synthonEditedVersion);
    }

    public SimpleMultiSynthonStructure getMultiStructure() {
        return multiStructure;
    }

    public StereoMolecule getSynthonEditedVersion() {
        return synthonEditedVersion;
    }

    /**
     * writes the synthon (edited version) into the original synthon
     */
    public void confirmChangeToSynthon() {
        this.synthonOriginalReference.clear();
        this.synthonEditedVersion.copyMolecule(this.synthonOriginalReference);
        this.synthonOriginalReference.ensureHelperArrays(Molecule.cHelperCIP);
        fireMultiSynthonStructureChanged();
    }

    public void revertEditedSynthon() {
        this.synthonEditedVersion.clear();
        this.synthonOriginalReference.copyMolecule(this.synthonEditedVersion);
    }

    public void reportChangeToEditedSynthon() {
        fireMultiSynthonStructureChanged();
    }

    public static interface MultiSynthonStructureEditorModelListener {
        public void multiSynthonStructureChanged();
    }

    private List<MultiSynthonStructureEditorModelListener> listeners = new ArrayList<>();

    public void addMultiSynthonStructureEditorModelListener(MultiSynthonStructureEditorModelListener listener) {
        listeners.add(listener);
    }

    public void removeMultiSynthonStructureEditorModelListener(MultiSynthonStructureEditorModelListener listener) {
        listeners.remove(listener);
    }

    private void fireMultiSynthonStructureChanged() {
        for (MultiSynthonStructureEditorModelListener listener : listeners) {
            listener.multiSynthonStructureChanged();
        }
    }

}
