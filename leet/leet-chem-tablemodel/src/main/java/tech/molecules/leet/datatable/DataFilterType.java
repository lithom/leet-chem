package tech.molecules.leet.datatable;

public interface DataFilterType<T> {

    public String getFilterName();

    /**
     * If true, then for instances of the filter, the
     *
     * @return
     */
    public boolean requiresInitialization();
    public DataFilter<T> createInstance(DataTableColumn<?,T> column);

}
