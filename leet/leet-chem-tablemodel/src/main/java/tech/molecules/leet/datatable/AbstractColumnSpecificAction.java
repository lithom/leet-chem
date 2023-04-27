package tech.molecules.leet.datatable;

import javax.swing.*;

public abstract class AbstractColumnSpecificAction<U> extends AbstractAction {

    private DataTableColumn<?,U> column;

    public AbstractColumnSpecificAction(String name, Icon icon, DataTableColumn dc) {
        super(name,icon);
    }

    protected DataTableColumn<?,U> getColumn() {
        return this.column;
    }

}
