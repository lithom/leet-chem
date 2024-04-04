package tech.molecules.leet.chem.virtualspaces.gui;

public class BuildingBlockFile {
    private String filepath;
    private String format;
    private String structureFieldName;
    private String idFieldName;
    private int numRows;

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    private String status;

    public BuildingBlockFile(String filepath, String format, String structureFieldName, String idFieldName, int numRows) {
        this.filepath = filepath;
        //this.format = format;
        this.structureFieldName = structureFieldName;
        this.idFieldName = idFieldName;
        this.numRows = numRows;
        this.status = "Initializing";
        this.identifyFormat();
    }

    private void identifyFormat() {
        // Get the file name
        String fileName = this.filepath;

        // Find the position of the last dot (.)
        int dotIndex = fileName.lastIndexOf('.');
        // Extract the file extension
        String fileExtension = "";
        if (dotIndex >= 0 && dotIndex < fileName.length() - 1) {
            fileExtension = fileName.substring(dotIndex);
        }
        this.format = fileExtension;
    }

    public String getFilepath() {
        return filepath;
    }

    public void setFilepath(String filepath) {
        this.filepath = filepath;
    }

    public String getFormat() {
        return format;
    }

    public void setFormat(String format) {
        this.format = format;
    }

    public String getStructureFieldName() {
        return structureFieldName;
    }

    public void setStructureFieldName(String structureFieldName) {
        this.structureFieldName = structureFieldName;
    }

    public String getIdFieldName() {
        return idFieldName;
    }

    public void setIdFieldName(String idFieldName) {
        this.idFieldName = idFieldName;
    }

    public int getNumRows() {
        return numRows;
    }

    public void setNumRows(int numRows) {
        this.numRows = numRows;
    }
}
