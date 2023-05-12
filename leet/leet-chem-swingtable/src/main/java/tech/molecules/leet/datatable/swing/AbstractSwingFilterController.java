package tech.molecules.leet.datatable.swing;

import tech.molecules.leet.datatable.DataFilter;
import tech.molecules.leet.datatable.DataFilterType;

import javax.swing.*;
import java.util.List;

public abstract class AbstractSwingFilterController<T> {

    private DataFilterType<T> filter;

    public abstract JPanel getConfigurationPanel();
    public abstract List<Action> getActions();

}
