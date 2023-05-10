package tech.molecules.leet.datatable;

import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;

import java.awt.*;
import java.util.*;
import java.util.List;

public class DataTableSelectionModel {



    public DataTableSelectionModel() {
        this.initDataTableSelectionTypes();
    }

    public static class SelectionType {
        private String name;
        private Color color;
        public SelectionType(String name, Color color) {
            this.name = name;
            this.color = color;
        }
        public String getName() {
            return name;
        }
        public Color getColor() {
            return color;
        }
        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            SelectionType that = (SelectionType) o;
            return new EqualsBuilder().append(name, that.name).isEquals();
        }
        @Override
        public int hashCode() {
            return new HashCodeBuilder(17, 37).append(name).toHashCode();
        }
    }



    private Map<String,SelectionType> selectionTypeRegistry = new HashMap<>();
    public SelectionType getSelectionType(String name) {
        return this.selectionTypeRegistry.get(name);
    }

    public Set<String> getSelectionTypeRows(String selectionTypeName) {
        return this.selections.get(this.selectionTypeRegistry.get(selectionTypeName));
    }

    private void initDataTableSelectionTypes() {
        this.registerSelectionType(new SelectionType(SELECTION_TYPE_SELECTED,Color.red.brighter().brighter()));
        this.registerSelectionType(new SelectionType(SELECTION_TYPE_MOUSE_OVER,Color.cyan.darker()));
    }

    public static final String SELECTION_TYPE_SELECTED   = "selected";
    public static final String SELECTION_TYPE_MOUSE_OVER = "mouseOver";

    /**
     * returns false if type was already registered
     * @param type
     * @return
     */
    public boolean registerSelectionType(SelectionType type) {
        if(this.selectionTypeRegistry.containsKey(type.getName())) {
            return false;
        }
        this.selectionTypeRegistry.put(type.getName(),type);
        return true;
    }

    private Map<String, java.util.List<SelectionType>> selectionTypes  = new HashMap<>();
    private Map<String,Color>       highlightColors = new HashMap<>();

    private Map<SelectionType,Set<String>> selections = new HashMap<>();

    public void addSelectionTypeToRows(SelectionType c, Collection<String> rows) {
        for(String rowid : rows) {
            if (!selectionTypes.containsKey(rowid)) {
                this.selectionTypes.put(rowid, new ArrayList<>());
            }
            this.selectionTypes.get(rowid).add(c);
        }
        if(!selections.containsKey(c)) {selections.put(c,new HashSet<>());}
        selections.get(c).addAll(rows);
        fireSelectionChanged(rows);
    }

    public List<SelectionType> getSelectionTypesForRow(String row) {
        List<SelectionType> st = new ArrayList<>();
        for(Map.Entry<SelectionType,Set<String>> sti : selections.entrySet()) {
            if(sti.getValue().contains(row)) {st.add(sti.getKey());}
        }
        return st;
    }

    public void removeSelectionTypeFromRows(SelectionType c, Collection<String> rows) {
        for(String rowid : rows) {
            List<SelectionType> ci = selectionTypes.get(rowid);
            if(ci!=null) {
                ci.remove(c);
            }
        }
        if(selections.containsKey(c)) {
            selections.get(c).removeAll(rows);
        }
        fireSelectionChanged(rows);
    }


    public void resetSelectionForSelectionType(SelectionType c) {
        if(!selections.containsKey(c)) {selections.put(c,new HashSet<>());}
        Set<String> old_selection = selections.get(c);
        selections.get(c).clear();
        fireSelectionChanged(old_selection);
    }

    public static interface SelectionListener {
        public void selectionStatusChanged(Collection<String> rows);
    }

    private List<SelectionListener> listeners = new ArrayList<>();

    public void addSelectionListener(SelectionListener li) {
        this.listeners.add(li);
    }

    public boolean removeSelectionListener(SelectionListener li) {
        return this.listeners.remove(li);
    }

    private void fireSelectionChanged(Collection<String> rows) {
        for(SelectionListener li : listeners) {
            li.selectionStatusChanged(rows);
        }
    }

}
