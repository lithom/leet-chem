package tech.molecules.leet.datatable.swing.testA;

import tech.molecules.leet.datatable.DataFilterType;
import tech.molecules.leet.datatable.DataTable;
import tech.molecules.leet.datatable.DataTableColumn;
import tech.molecules.leet.datatable.swing.AbstractSwingFilterController;

import javax.swing.*;
import java.util.List;

public class RegExpFilterController extends AbstractSwingFilterController<String> {

    public RegExpFilterController(DataTable dt, DataTableColumn<?, String> column, DataFilterType<String> filter) {
        super(dt, column, filter);
    }

    @Override
    public JPanel getConfigurationPanel() {
        return null;
    }

}
