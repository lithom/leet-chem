package tech.molecules.chem.coredb;

import java.util.Date;
import java.util.List;

public class AssayResultQuery {
    private Integer assayId;
    private Date fromDate;
    private Date toDate;
    private List<String> compoundIds;

    // Getters and setters for each parameter

    public AssayResultQuery(Integer assayId, Date fromDate, Date toDate, List<String> compoundIds) {
        this.assayId = assayId;
        this.fromDate = fromDate;
        this.toDate = toDate;
        this.compoundIds = compoundIds;
    }

    public Integer getAssayId() {
        return assayId;
    }

    public void setAssayId(Integer assayId) {
        this.assayId = assayId;
    }

    public Date getFromDate() {
        return fromDate;
    }

    public void setFromDate(Date fromDate) {
        this.fromDate = fromDate;
    }

    public Date getToDate() {
        return toDate;
    }

    public void setToDate(Date toDate) {
        this.toDate = toDate;
    }

    public List<String> getCompoundIds() {
        return compoundIds;
    }

    public void setCompoundIds(List<String> compoundIds) {
        this.compoundIds = compoundIds;
    }
}

