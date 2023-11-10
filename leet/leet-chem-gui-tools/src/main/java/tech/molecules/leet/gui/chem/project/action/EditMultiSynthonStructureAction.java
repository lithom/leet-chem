package tech.molecules.leet.gui.chem.project.action;

import com.actelion.research.chem.StereoMolecule;
import tech.molecules.leet.chem.sar.SimpleMultiSynthonStructure;
import tech.molecules.leet.chem.sar.SimpleSARSeries;
import tech.molecules.leet.gui.chem.editor.sar.MultiSynthonStructureEditorModel;
import tech.molecules.leet.gui.chem.editor.sar.MultiSynthonStructureEditorView;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.util.function.Consumer;
import java.util.function.Supplier;

public class EditMultiSynthonStructureAction extends AbstractAction implements ObjectSpecific {

    private final Consumer<StereoMolecule> callbackMultiStructureChanged;
    private final Supplier<JPanel> targetPanel;

    private SimpleMultiSynthonStructure multiStructure;

    public EditMultiSynthonStructureAction(Consumer<StereoMolecule> callbackMultiStructureChanged, Supplier<JPanel> targetPanel) {
        super("Edit Structures..");
        this.callbackMultiStructureChanged = callbackMultiStructureChanged;
        this.targetPanel = targetPanel;
        //this.multiStructure = multiStructure;
    }



    @Override
    public void setObject(Object obj) {
        if(obj instanceof SimpleMultiSynthonStructure) {
            this.multiStructure = (SimpleMultiSynthonStructure) obj;
        }
        if(obj instanceof SimpleSARSeries) {
            this.multiStructure = ((SimpleSARSeries)obj).getSeriesDecomposition();
        }
    }

    @Override
    public Object getObject() {
        return this.multiStructure;
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        // Create dialog components
        JPanel panel = this.targetPanel.get();

        panel.removeAll();
        panel.setLayout(new BorderLayout());

        // create the synthon editor
        MultiSynthonStructureEditorModel model = new MultiSynthonStructureEditorModel(multiStructure,multiStructure.getSynthonSets().get(0).getSynthons().get(0),this.callbackMultiStructureChanged);

        MultiSynthonStructureEditorView view = new MultiSynthonStructureEditorView(model);
        panel.add(view,BorderLayout.CENTER);

    }

}
