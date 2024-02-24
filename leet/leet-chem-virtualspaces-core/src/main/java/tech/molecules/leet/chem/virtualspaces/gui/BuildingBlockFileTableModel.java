package tech.molecules.leet.chem.virtualspaces.gui;

import javax.swing.table.AbstractTableModel;
import java.util.*;

public class BuildingBlockFileTableModel extends AbstractTableModel {
    private final List<BuildingBlockFile> buildingBlockFiles;

    private final int colFilepath = 0;
    private final int colFormat = 1;
    private final int colStructureFieldName = 2;
    private final int colIDFieldName = 3;
    private final int colStatus = 4;
    private final int colSelected = 5;

    private Map<BuildingBlockFile,List<LoadedBB>> loadedBBs = new HashMap<>();

    private final String[] columnNames = {"Filepath", "Format", "Structure Field Name", "ID Field Name", "Status", "Selected"};
    private final Class[] columnClasses = {String.class, String.class, String.class, String.class, String.class, Boolean.class};

    private Set<BuildingBlockFile> selectedBuildingBlockFiles = new HashSet<>();

    public BuildingBlockFileTableModel(List<BuildingBlockFile> buildingBlockFiles) {
        this.buildingBlockFiles = buildingBlockFiles;
    }

    public void updateStatus(BuildingBlockFile bbf) {
        int bidx = this.buildingBlockFiles.indexOf(bbf);
        this.fireTableRowsUpdated(bidx,bidx);
    }

    @Override
    public int getRowCount() {
        return buildingBlockFiles.size();
    }

    @Override
    public int getColumnCount() {
        return columnNames.length;
    }

    @Override
    public Class<?> getColumnClass(int columnIndex) {
        return columnClasses[columnIndex];
    }

    @Override
    public Object getValueAt(int rowIndex, int columnIndex) {
        BuildingBlockFile file = buildingBlockFiles.get(rowIndex);
        switch (columnIndex) {
            case 0: return file.getFilepath();
            case 1: return file.getFormat();
            case 2: return file.getStructureFieldName();
            case 3: return file.getIdFieldName();
            case 4: return file.getStatus();//return file.get;
            case 5: return this.selectedBuildingBlockFiles.contains(file);
            default: throw new IllegalArgumentException("Invalid column index");
        }
    }

    @Override
    public boolean isCellEditable(int rowIndex, int columnIndex) {
        // Only make "Structure Field Name", "ID Field Name", and "Selected" columns editable
        return columnIndex == colStructureFieldName || columnIndex == colIDFieldName || columnIndex == colSelected; // Assuming "Selected" is at index 5
    }

    @Override
    public void setValueAt(Object aValue, int rowIndex, int columnIndex) {
        BuildingBlockFile file = buildingBlockFiles.get(rowIndex);
        switch (columnIndex) {
            case 2: // Structure Field Name
                file.setStructureFieldName((String) aValue);
                break;
            case 3: // ID Field Name
                file.setIdFieldName((String) aValue);
                break;
            case 5: // Selected
                if((Boolean)aValue) {
                    this.selectedBuildingBlockFiles.add(file);
                }
                else {
                    this.selectedBuildingBlockFiles.remove(file);
                }
                break;
            default:
                break;
        }
        fireTableCellUpdated(rowIndex, columnIndex);

        if (columnIndex == 2 || columnIndex == 3) { // If editable fields change
            startAnalysisWorker(rowIndex); // Start background analysis
        }
    }

    public void startAnalysisWorker(int rowIndex) {
        new BuildingBlockFileAnalysisWorker(this,this.buildingBlockFiles.get(rowIndex)).execute();
    }

    @Override
    public String getColumnName(int column) {
        return columnNames[column];
    }

    public void addBuildingBlockFile(BuildingBlockFile file) {
        buildingBlockFiles.add(file);
        this.selectedBuildingBlockFiles.add(file);
        fireTableRowsInserted(buildingBlockFiles.size() - 2, buildingBlockFiles.size() - 1);
    }

    public void updateBuildingBlockFiles() {
        fireTableCellUpdated(0,this.getRowCount());
    }

    public void removeBuildingBlockFile(int rowIndex) {
        buildingBlockFiles.remove(rowIndex);
        fireTableRowsDeleted(rowIndex, rowIndex);
    }

    public void setLoadedBBs(BuildingBlockFile bbFile, List<LoadedBB> loadedBBs) {
        this.loadedBBs.put(bbFile,loadedBBs);
    }
}
