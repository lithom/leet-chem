package tech.molecules.leet.datatable.swing.chem;

import com.actelion.research.chem.StereoMolecule;
import tech.molecules.leet.chem.StructureRecord;
import tech.molecules.leet.datatable.DataFilter;
import tech.molecules.leet.datatable.DataTable;
import tech.molecules.leet.datatable.DataTableColumn;
import tech.molecules.leet.datatable.chem.SubstructureFilter;
import tech.molecules.leet.datatable.swing.FilterActionProvider;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.util.function.Consumer;
import java.util.function.Supplier;

public class FilterActionProviderSubstructure implements FilterActionProvider {

    private Supplier<JPanel> targetPanel;

    public FilterActionProviderSubstructure(Supplier<JPanel> targetPanel) {
        this.targetPanel = targetPanel;
    }

    @Override
    public Action getAddFilterAction(DataTable dtable, DataTableColumn dcol, DataFilter filter) {
        if(filter instanceof SubstructureFilter) {
            SubstructureFilter sfilter = (SubstructureFilter) filter;
            return new AddSubstructureFilterAction(dtable,dcol, (xi) -> sfilter.setSubstructure(xi),this.targetPanel);
        }
        return null;
    }

//    @Override
//    public List<Action> getFilterActions() {
//        List<Action> actions = new ArrayList<>();
//        actions.add(new AddSubstructureFilterAction(dtable,dcol,callback,targetPanel));
//        return actions;
//    }


    public static class AddSubstructureFilterAction extends AbstractAction {

        private DataTable dtable;
        private DataTableColumn<?,? extends StructureRecord> col;
        private Consumer<StereoMolecule> callback;
        private Supplier<JPanel> targetPanel;

        public AddSubstructureFilterAction(DataTable dt,
                                           DataTableColumn<?, ? extends StructureRecord> col,
                                           Consumer<StereoMolecule> callback,
                                           Supplier<JPanel> targetPanel) {
            super("Add Substructure Filter");
            this.dtable = dt;
            this.col = col;
            this.callback = callback;
            this.targetPanel = targetPanel;
        }

        @Override
        public void actionPerformed(ActionEvent e) {
            SubstructureFilter sf = new SubstructureFilter();
            SubstructureFilterController fc = new SubstructureFilterController(dtable,col,sf);
            JPanel pi = targetPanel.get();
            pi.removeAll();
            pi.setLayout(new BorderLayout());
            pi.add(fc.getConfigurationPanel(),BorderLayout.CENTER);
            pi.getParent().validate();
            pi.getParent().repaint();
            dtable.addFilter(this.col,sf);
        }
    }

}
