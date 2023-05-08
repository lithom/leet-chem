package tech.molecules.leet.datatable.swing;

import org.apache.commons.lang3.tuple.Pair;
import tech.molecules.leet.datatable.DataRepresentation;

import javax.swing.table.TableCellRenderer;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class SwingDataTableRendererRegistry {

    private static SwingDataTableRendererRegistry INSTANCE;

    public static SwingDataTableRendererRegistry getInstance() {
        if(INSTANCE == null) {
            INSTANCE = new SwingDataTableRendererRegistry();
        }
        return INSTANCE;
    }

    /**
     * Map from original class to alternative representations
     * access synchronized via "this"
     */
    private Map<Class, List<TableCellRenderer>> renderers;

    public synchronized <T> void registerRenderer(Class<T> original, TableCellRenderer renderer) {
        if(!renderers.containsKey(original)) {
            renderers.put(original,new ArrayList<>());}
        renderers.get(original).add(renderer);
    }

    public synchronized List<TableCellRenderer> findRenderers(Class original) {
        return renderers.get(original);
    }

}
