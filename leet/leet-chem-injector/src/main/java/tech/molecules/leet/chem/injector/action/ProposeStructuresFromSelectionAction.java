package tech.molecules.leet.chem.injector.action;

import com.actelion.research.chem.StereoMolecule;
import tech.molecules.leet.chem.injector.Injector;
import tech.molecules.leet.chem.injector.InjectorTools;
import tech.molecules.leet.chem.mutator.FragmentDecompositionSynthon;
import tech.molecules.leet.chem.mutator.SynthonWithContext;
import tech.molecules.leet.chem.shredder.FragmentDecomposition;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.util.List;
import java.util.function.Supplier;

public class ProposeStructuresFromSelectionAction extends AbstractAction {

    private Supplier<StereoMolecule> molWithSelection;

    public ProposeStructuresFromSelectionAction(Supplier<StereoMolecule> molWithSelection) {
        super("Propose structures from selection");
        this.molWithSelection = molWithSelection;
    }

    @Override
    public void actionPerformed(ActionEvent e) {

        StereoMolecule m_with_selection = this.molWithSelection.get();//Meditor.getSwingEditorPanel().getSwingDrawArea().getGenericDrawArea().getMolecule();

        FragmentDecomposition fd = null;
        try {
            fd = InjectorTools.computeFragmentDecompositionFromSelection(m_with_selection);
        } catch (InjectorTools.InvalidSelectionException ex) {
            throw new RuntimeException(ex);
        }

        SynthonWithContext swc = new FragmentDecompositionSynthon(fd);
        System.out.println("fd: "+fd.toString());
        System.out.println("swc: "+swc.toString());


        StereoMolecule context_bd1 = swc.getContextBidirectirectional(1,1);
        System.out.println( context_bd1.getIDCode() );
        //System.out.println("compute assemblies:");

        //System.out.println( swc.getContextBidirectirectional(1,1).getIDCode() );

        List<String> synthons = Injector.INJECTOR.getSimpleSynthonsForContext(context_bd1.getIDCode());
        System.out.println("synthons: "+synthons.size());
    }
}
