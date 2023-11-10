package tech.molecules.leet.datatable;

import java.util.ArrayList;
import java.util.List;

/**
 *
 * @param <U> class of the dataprovider / column
 * @param <C> class data can specify more precisely the meaning of the class
 */
public interface CategoricDatasource<U,C> extends DataRepresentation<U,String> {

    public DataTableColumn<?,U> getColumn();

    /**
     * @param categoryString
     * @return
     */
    public C getCategory(String categoryString);

    default List<String> getCategoryVisibleColumns(DataTable table) {
        List<String> rows = table.getVisibleKeysSorted();
        List<String> data = new ArrayList<>();
        DataTableColumn<?,U> ci = getColumn();
        for(String si : rows) {
            data.add( evaluate( ci.getRawValue(si) ) );
        }
        return data;
    }
}
