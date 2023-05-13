package tech.molecules.leet.datatable.swing;

import tech.molecules.leet.datatable.DataFilter;
import tech.molecules.leet.datatable.DataFilterType;
import tech.molecules.leet.datatable.DataTable;
import tech.molecules.leet.datatable.DataTableColumn;

import javax.swing.*;
import java.util.List;

public abstract class AbstractSwingFilterController<T> {

    protected DataTable dtable;
    protected DataTableColumn<?,T> column;
    protected DataFilterType<T> filter;

    public AbstractSwingFilterController(DataTable dt, DataTableColumn<?, T> column, DataFilterType<T> filter) {
        this.dtable = dt;
        this.column = column;
        this.filter = filter;
    }

    public abstract JPanel getConfigurationPanel();



}
