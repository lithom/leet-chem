package tech.molecules.leet.table;

import javax.swing.*;
import javax.swing.table.TableCellEditor;
import java.awt.*;

public class DefaultStringCellEditor  extends AbstractCellEditor implements TableCellEditor {
    private Object lastValue = null;

    private JPanel jp_editor = new JPanel();

    private NColumn column;

    public DefaultStringCellEditor(NColumn col) {
        this.column = col;
    }

    private Color c_BG_a = Color.WHITE;
    private Color c_BG_b = new Color(248, 252, 252);


    @Override
    public Component getTableCellEditorComponent(JTable table, Object value, boolean isSelected, int row, int column) {
        this.lastValue = value;

        this.jp_editor.setOpaque(false);
        //this.jp_editor.setSize(table.getColumnModel().getColumn(column).getWidth(),table.getRowHeight(row));
        //this.jp_editor.setBackground((row%2==0)?c_BG_a:c_BG_a);

        this.jp_editor.removeAll();

        if (value == null) {
            this.jp_editor.removeAll();
        }

        if (value instanceof String) {
            this.jp_editor.removeAll();
            this.jp_editor.setLayout(new FlowLayout());
            String str_a = (String)value;
            this.jp_editor.add(new DefaultNumericCellEditor.JEditorLabel(str_a));
        }

        return this.jp_editor;
    }

    @Override
    public Object getCellEditorValue() {
        return lastValue;
    }
}