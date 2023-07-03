package tech.molecules.leet.gui.chem.project.action;

import com.actelion.research.chem.StereoMolecule;
import com.actelion.research.chem.combinatorialspace.Synthon;
import tech.molecules.leet.chem.sar.SimpleMultiSynthonStructure;
import tech.molecules.leet.chem.sar.SimpleSARSeries;
import tech.molecules.leet.chem.sar.SimpleSynthonSet;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

public class AddSynthonSetAction extends AbstractAction implements ObjectSpecific {

    private Consumer<Object> callbackMultiStructureChanged;
    private SimpleMultiSynthonStructure synthonSetConsumer;

    public AddSynthonSetAction(Consumer<Object> callbackMultiStructureChanged) {
        super("Add Synthon Set");
        this.callbackMultiStructureChanged = callbackMultiStructureChanged;
    }

    @Override
    public void setObject(Object obj) {
        if(obj instanceof SimpleMultiSynthonStructure) {
            this.synthonSetConsumer = (SimpleMultiSynthonStructure) obj;
        }
    }

    @Override
    public Object getObject() {
        return this.synthonSetConsumer;
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        List<StereoMolecule> mols_a = new ArrayList<>();
        mols_a.add(new StereoMolecule());
        SimpleSynthonSet ssa = new SimpleSynthonSet(mols_a);
        this.synthonSetConsumer.getSynthonSets().add(ssa);
        if(callbackMultiStructureChanged!=null) {
            callbackMultiStructureChanged.accept(new Object());
        }
    }
}
