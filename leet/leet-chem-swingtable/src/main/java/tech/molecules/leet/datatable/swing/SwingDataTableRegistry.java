package tech.molecules.leet.datatable.swing;

import javax.swing.table.TableCellRenderer;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class SwingDataTableRegistry {

    private static SwingDataTableRegistry INSTANCE;

    public static SwingDataTableRegistry getInstance() {
        if(INSTANCE == null) {
            INSTANCE = new SwingDataTableRegistry();
        }
        return INSTANCE;
    }

    /**
     * Map from original class to avilable renderers
     * access synchronized via "this"
     */
    private Map<Class, List<TableCellRenderer>> renderers;

    /**
     * Map from original class to available filters
     * access synchronized via "this"
     */
    private Map<Class, List<AbstractSwingFilterController>> filter;

    public synchronized <T> void registerRenderer(Class<T> original, TableCellRenderer renderer) {
        if(!renderers.containsKey(original)) {
            renderers.put(original,new ArrayList<>());}
        renderers.get(original).add(renderer);
    }

    public synchronized List<TableCellRenderer> findRenderers(Class original) {
        return renderers.get(original);
    }




}
