package tech.molecules.leet.datatable.swing;

import com.actelion.research.gui.table.ChemistryCellRenderer;
import tech.molecules.leet.chem.StructureWithID;
import tech.molecules.leet.datatable.DataTableColumn;

import javax.swing.*;
import javax.swing.table.DefaultTableCellRenderer;
import java.awt.*;

public class CellRendererHelper {

    public static void configureDefaultRenderers(DefaultSwingTableController table) {
        for(int zi=0;zi<table.getModel().getDataTable().getDataColumns().size();zi++) {
            DataTableColumn dtc = table.getModel().getDataTable().getDataColumns().get(zi);
            if( dtc.getRepresentationClass().isAssignableFrom(StructureWithID.class)) {
                table.setTableCellRenderer(zi,new ChemistryCellRenderer() {
                    @Override
                    public Component getTableCellRendererComponent(JTable table, Object value, boolean isSelected, boolean hasFocus, int row, int col) {
                        String[] val_a = null;
                        if(value instanceof DataTableColumn.CellValue) {
                            DataTableColumn.CellValue value_cv = (DataTableColumn.CellValue) value;
                            if( value_cv.val instanceof StructureWithID) {
                                val_a = ((StructureWithID) value_cv.val).structure;
                            }
                        }
                        return super.getTableCellRendererComponent(table, val_a[0]+" "+val_a[1], isSelected, hasFocus, row, col);
                    }
                });
            }
            else if(dtc.getRepresentationClass().isAssignableFrom(Double.class)) {
                table.setTableCellRenderer(zi, new DefaultTableCellRenderer() {
                    @Override
                    public Component getTableCellRendererComponent(JTable table, Object value, boolean isSelected, boolean hasFocus, int row, int column) {
                        if (value instanceof DataTableColumn.CellValue) {
                            DataTableColumn.CellValue cv = (DataTableColumn.CellValue) value;
                            if (cv.val != null) {
                                this.setText(cv.val.toString());
                            } else {
                                this.setText("");
                            }
                        }
                        return this;
                    }
                });
            } else {
                table.setTableCellRenderer(zi, new DefaultTableCellRenderer() {
                    @Override
                    public Component getTableCellRendererComponent(JTable table, Object value, boolean isSelected, boolean hasFocus, int row, int column) {
                        if (value instanceof DataTableColumn.CellValue) {
                            DataTableColumn.CellValue cv = (DataTableColumn.CellValue) value;
                            if (cv.val != null) {
                                this.setText(cv.val.toString());
                            } else {
                                this.setText("");
                            }
                        }
                        return this;
                    }
                });
            }
        }
    }

}
