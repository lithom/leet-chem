package tech.molecules.leet.datatable.swing.chem;

import com.actelion.research.chem.StereoMolecule;
import com.actelion.research.gui.JEditableStructureView;
import com.actelion.research.gui.StructureListener;
import tech.molecules.leet.chem.StructureRecord;
import tech.molecules.leet.datatable.DataFilterType;
import tech.molecules.leet.datatable.DataTable;
import tech.molecules.leet.datatable.DataTableColumn;
import tech.molecules.leet.datatable.chem.SubstructureFilter;
import tech.molecules.leet.datatable.swing.AbstractSwingFilterController;

import javax.swing.*;
import java.awt.*;
import java.util.function.Consumer;

public class SubstructureFilterController<T extends StructureRecord> extends AbstractSwingFilterController<T> {

    private SubstructureFilter<T> filter;

    // GUI:
    private JPanel panel;
    private JEditableStructureView structureView;

    //private Consumer<StereoMolecule> callbackStructureChanged;

    public SubstructureFilterController(DataTable dt, DataTableColumn<?,T> column, SubstructureFilter<T> filter) {
        super(dt, column, filter.getDataFilterType());
        //this.callbackStructureChanged = callbackStructureChanged;
        this.filter = filter;
        reinit();
    }

    private void reinit() {
        if(this.panel == null) {
            this.panel = new JPanel();
        }
        this.panel.removeAll();
        this.panel.setLayout(new BorderLayout());
        if(this.structureView==null) {
            StereoMolecule fi = new StereoMolecule();
            fi.setFragment(true);
            this.structureView = new JEditableStructureView(fi);
            this.structureView.setAllowQueryFeatures(true);
            this.structureView.repaint();
        }
        this.panel.add(this.structureView,BorderLayout.CENTER);

        this.structureView.addStructureListener(new StructureListener() {
            @Override
            public void structureChanged(StereoMolecule stereoMolecule) {
                filter.setSubstructure(structureView.getMolecule());
            }
        });
    }

    @Override
    public JPanel getConfigurationPanel() {
        return this.panel;
    }



}
