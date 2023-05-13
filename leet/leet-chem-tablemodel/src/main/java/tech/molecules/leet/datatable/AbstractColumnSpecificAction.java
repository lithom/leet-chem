package tech.molecules.leet.datatable;

import javax.swing.*;

public abstract class AbstractColumnSpecificAction<U> {

    private String name;
    private DataTableColumn<?,U> column;

    public AbstractColumnSpecificAction(String name, DataTableColumn dc) {
        this.name = name;
        this.column = dc;
    }

    public abstract void run();

    protected DataTableColumn<?,U> getColumn() {
        return this.column;
    }

}
