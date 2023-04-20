package tech.molecules.chem.coredb.gui;
import tech.molecules.chem.coredb.Assay;
import tech.molecules.chem.coredb.AssayQuery;
import tech.molecules.chem.coredb.sql.DBManager;

import javax.swing.table.AbstractTableModel;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class AssayTableModel extends AbstractTableModel {
    private DBManager dbManager;
    private List<Assay> assays;
    private List<Assay> filteredAssays;
    private String[] columnNames = {"Assay Name", "Assay ID", "Project", "Number of Measurements"};

    private Map<Assay,Integer> numberOfResultsCache = new HashMap<>();

    public AssayTableModel(DBManager dbManager) {
        this.dbManager = dbManager;
        this.assays = new ArrayList<>();
        this.filteredAssays = new ArrayList<>();
    }

    public void loadData() throws SQLException {
        assays = dbManager.searchAssays(new AssayQuery());
        filteredAssays = new ArrayList<>(assays);
        fireTableDataChanged();
    }

    public void filterData(String searchString) {
        if (searchString == null || searchString.isEmpty()) {
            filteredAssays = new ArrayList<>(assays);
        } else {
            String fSearchString = searchString.toLowerCase();
            filteredAssays = assays.stream()
                    .filter(assay -> assay.getName().toLowerCase().contains(fSearchString) || assay.getProject().getName().toLowerCase().contains(searchString))
                    .collect(Collectors.toList());
        }
        fireTableDataChanged();
    }

    @Override
    public int getRowCount() {
        return filteredAssays.size();
    }

    @Override
    public int getColumnCount() {
        return columnNames.length;
    }

    @Override
    public String getColumnName(int column) {
        return columnNames[column];
    }

    @Override
    public Object getValueAt(int rowIndex, int columnIndex) {
        Assay assay = filteredAssays.get(rowIndex);
        switch (columnIndex) {
            case 0:
                return assay.getName();
            case 1:
                return assay.getId();
            case 2:
                return assay.getProject().getName();
            case 3:
                if(!numberOfResultsCache.containsKey(assay)) {
                    try {
                        numberOfResultsCache.put(assay,-1);
                        numberOfResultsCache.put( assay , dbManager.getNumberOfMeasurements(assay) );
                    } catch (SQLException e) {
                        e.printStackTrace();
                        return 0;
                    }
                }
                return numberOfResultsCache.get(assay);
            default:
                return null;
        }
    }
}
