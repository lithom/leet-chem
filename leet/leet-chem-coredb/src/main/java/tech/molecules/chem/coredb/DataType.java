package tech.molecules.chem.coredb;

public enum DataType {
    NUMERIC("Numeric"),
    ALPHANUMERIC("Alphanumeric"),
    TEXT("TEXT");

    private final String value;

    private DataType(String value) {
        this.value = value;
    }

    public String getValue() {
        return value;
    }

    public static DataType fromValue(String value) {
        for (DataType dataType : DataType.values()) {
            if (dataType.getValue().equalsIgnoreCase(value)) {
                return dataType;
            }
        }
        throw new IllegalArgumentException("No matching DataType for value: " + value);
    }

}