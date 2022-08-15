package tech.molecules.leet.table;

import javax.swing.*;
import javax.swing.border.LineBorder;
import javax.swing.table.TableCellEditor;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.List;

public class DefaultNumericCellEditor extends AbstractCellEditor implements TableCellEditor {
    private Object lastValue = null;

    private JPanel jp_editor = new JPanel();

    private String formatNumeric = "%.3f";

    private NColumn column;

    public DefaultNumericCellEditor(NColumn col) {
        this.column = col;
    }

    private Color c_BG_a = Color.WHITE;
    private Color c_BG_b = new Color(248,252,252);


    @Override
    public Component getTableCellEditorComponent(JTable table, Object value, boolean isSelected, int row, int column) {
        this.lastValue = value;

        this.jp_editor.setOpaque(false);
        //this.jp_editor.setSize(table.getColumnModel().getColumn(column).getWidth(),table.getRowHeight(row));
        //this.jp_editor.setBackground((row%2==0)?c_BG_a:c_BG_a);

        if (value == null) {
            this.jp_editor.removeAll();
        }

        if(value instanceof Double) {
            this.jp_editor.removeAll();
            this.jp_editor.setLayout(new FlowLayout());
            String str_a = "";
            double evaluatedNumber = (double) value;

            str_a = String.format(formatNumeric, evaluatedNumber);
            this.jp_editor.add(new JEditorLabel(str_a));
        }
        if(value instanceof double[]) {
            double[] dav = (double[]) value;
            this.jp_editor.removeAll();
            this.jp_editor.setLayout(new FlowLayout());
            for(int zi=0;zi<dav.length;zi++) {
                String str_a = "";
                double evaluatedNumber = dav[zi];

                str_a = String.format(formatNumeric, evaluatedNumber);
                this.jp_editor.add(new JEditorLabel(str_a));
            }
        }

        return this.jp_editor;
    }

    @Override
    public Object getCellEditorValue() {
        return lastValue;
    }

    public static class JEditorLabel extends JLabel {
        public JEditorLabel(String label) {
            super(label);

            setBackground(new Color(252,252,252));
            setOpaque(false);

            this.setBorder(new LineBorder(new Color(0,0,0,0),1));
            this.addMouseListener(new MouseAdapter() {
                @Override
                public void mouseEntered(MouseEvent e) {
                    super.mouseEntered(e);
                    setBorder(new LineBorder(Color.blue,1));
                }
                @Override
                public void mouseExited(MouseEvent e) {
                    super.mouseExited(e);
                    setBorder(new LineBorder(new Color(0,0,0,0),1));

                }
            });
        }
    }

}
