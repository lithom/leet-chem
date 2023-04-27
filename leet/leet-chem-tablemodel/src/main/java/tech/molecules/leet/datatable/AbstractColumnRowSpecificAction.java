package tech.molecules.leet.datatable;

import javax.swing.*;
import java.util.List;

public abstract class AbstractColumnRowSpecificAction<U> extends AbstractAction {

    private DataTableColumn<?,U> column;
    private List<String> selectedRows;

    public AbstractColumnRowSpecificAction(String name, Icon icon, DataTableColumn dc, List<String> selectedRows) {
        super(name,icon);
        this.column = dc;
        this.selectedRows = selectedRows;
    }

    protected DataTableColumn<?,U> getColumn() {
        return this.column;
    }

    protected List<String> getSelectedRows() {
        return this.selectedRows;
    }

}

