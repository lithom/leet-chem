package tech.molecules.leet.table.gui;

import javax.swing.*;
import javax.swing.table.TableCellEditor;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;

public class InteractiveJTable extends JTable {

    public interface TwoStageTableCellEditor extends TableCellEditor
    {
        public boolean isFullyEngaged();
        public boolean isInStageTwo();
    }

    public InteractiveJTable() {

    }

    // In the body of your JTable subclass...
    private final MouseAdapter twoStageEditingListener = new MouseAdapter() {
        public void mouseMoved(MouseEvent e) {
            possiblySwitchEditors(e);
        }
        public void mouseEntered(MouseEvent e) {
            possiblySwitchEditors(e);
        }
        public void mouseExited(MouseEvent e) {
            possiblySwitchEditors(e);
        }
        public void mouseClicked(MouseEvent e) {
            possiblySwitchEditors(e);
        }
    };
    private void possiblySwitchEditors(MouseEvent e) {
        Point p = e.getPoint();
        if (p != null) {
            int row = rowAtPoint(p);
            int col = columnAtPoint(p);
            if (row != getEditingRow() || col != getEditingColumn()) {
                if (isEditing()) {
                    TableCellEditor editor = getCellEditor();
                    if (editor instanceof TwoStageTableCellEditor && !((TwoStageTableCellEditor)editor).isInStageTwo()) {
                        if (!editor.stopCellEditing()) {
                            editor.cancelCellEditing();
                        }
                    }
                }

                if (!isEditing()) {
                    if (row != -1 && isCellEditable(row, col)) {
                        editCellAt(row, col);
                    }
                }
            }
        }
    }


}
