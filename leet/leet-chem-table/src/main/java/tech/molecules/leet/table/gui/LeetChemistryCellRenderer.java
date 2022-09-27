package tech.molecules.leet.table.gui;

import com.actelion.research.chem.*;
import com.actelion.research.chem.coords.CoordinateInventor;
import com.actelion.research.chem.reaction.Reaction;
import com.actelion.research.chem.reaction.ReactionEncoder;
import com.actelion.research.gui.generic.GenericDepictor;
import com.actelion.research.gui.generic.GenericDrawContext;
import com.actelion.research.gui.generic.GenericRectangle;
import com.actelion.research.gui.hidpi.HiDPIHelper;
import com.actelion.research.gui.swing.SwingDrawContext;
import com.actelion.research.gui.table.ChemistryRenderPanel;
import tech.molecules.leet.table.NexusTable;

import javax.swing.*;
import javax.swing.table.TableCellEditor;
import javax.swing.table.TableCellRenderer;
import java.awt.*;

public class LeetChemistryCellRenderer extends AbstractCellEditor implements TableCellEditor, ListCellRenderer, TableCellRenderer {
    private MyChemistryRenderPanel  mRenderPanel;
    private boolean					mIsEnabled,mAlternateBackground;

    public LeetChemistryCellRenderer() {
        this(null);
    }

    public LeetChemistryCellRenderer(Dimension preferredSize) {
        mRenderPanel = new MyChemistryRenderPanel();
        if (preferredSize != null)
            mRenderPanel.setPreferredSize(preferredSize);
    }

    public void setAlternateRowBackground(boolean b) {
        mAlternateBackground = b;
    }

    private Object lastValue = null;

    @Override
    public Component getTableCellEditorComponent(JTable table, Object value, boolean isSelected, int row, int column) {
        this.lastValue = value;
        return getTableCellRendererComponent(table,value,isSelected,false,row,column);
    }

    @Override
    public Object getCellEditorValue() {
        return lastValue;
    }

    public Component getListCellRendererComponent(JList list, Object value, int index, boolean isSelected, boolean hasFocus) {
        mIsEnabled = list.isEnabled();
        return getCellRendererComponent(null, value, isSelected, hasFocus, index);
    }

    public Component getTableCellRendererComponent(JTable table, Object value, boolean isSelected, boolean hasFocus,int row, int col) {
        mIsEnabled = table.isEnabled();
        return getCellRendererComponent(table, value, isSelected, hasFocus, row);
    }

    private Component getCellRendererComponent(JTable table, Object value, boolean isSelected, boolean hasFocus, int row) {
//        if (LookAndFeelHelper.isAqua()
//                || LookAndFeelHelper.isQuaQua())
//            mRenderPanel.setOpaque(true);
//        else
//            mRenderPanel.setOpaque(false);
        JPanel pi = null;
        if(table instanceof NexusTable) {
            NexusTable nt = ((NexusTable) table);
            //System.out.println("mkay");
            pi = NexusTable.getDefaultEditorBackgroundPanel(nt,nt.getTableModel().getHighlightingAndSelectionStatus(row));
        }
        else {
            pi = new JPanel();
        }
        pi.setLayout(new BorderLayout());
        pi.add(mRenderPanel,BorderLayout.CENTER);

        mRenderPanel.setOpaque(false);

        if (value == null) {
            mRenderPanel.setChemistry(null);
        }
        else if (value instanceof String) {
            String s = (String)value;
            if (s.length() == 0) {
                mRenderPanel.setChemistry(null);
            }
            else {
                // If we have a PRODUCT_IDENTIFIER we have a reaction,
                // unless we have an idcode+SPACE+coords with coords starting with PRODUCT_IDENTIFIER.
                int productIndex = s.indexOf(ReactionEncoder.PRODUCT_IDENTIFIER);
                if (productIndex > 0 && s.charAt(productIndex-1) == ' ')
                    productIndex = -1;

                if (productIndex != -1) {
                    mRenderPanel.setChemistry(ReactionEncoder.decode((String)value, true));
                }
                else {
                    int index = s.indexOf('\n');
                    if (index == -1) {
                        index = s.indexOf(' ');
                        if (index == -1)
                            mRenderPanel.setChemistry(new IDCodeParser(true).getCompactMolecule(s));
                        else
                            mRenderPanel.setChemistry(new IDCodeParser(true).getCompactMolecule(
                                    s.substring(0, index),
                                    s.substring(index+1)));
                    }
                    else {
                        StereoMolecule mol = new StereoMolecule();
                        new IDCodeParser(true).parse(mol, s.substring(0, index));
                        do {
                            s = s.substring(index+1);
                            index = s.indexOf('\n');
                            mol.addMolecule(new IDCodeParser(true).getCompactMolecule(index == -1 ? s : s.substring(0, index)));
                        } while (index != -1);
                        new CoordinateInventor().invent(mol);
                        mRenderPanel.setChemistry(mol);
                    }
                }
            }
        }
        else {
            mRenderPanel.setChemistry(value);
        }

        //mRenderPanel.setAlternateBackground(mAlternateBackground && (row & 1) == 1);
        //mRenderPanel.setSelected(isSelected);
        //mRenderPanel.setFocus(hasFocus);
        //if(!mIsEnabled) {
        //    mRenderPanel.setOverruleForeground(Color.GRAY);
        //}
        //mRenderPanel.setOverruleForeground(mIsEnabled ? null : Color.GRAY);
        //return mRenderPanel;
        return pi;
    }


    private class MyChemistryRenderPanel extends ChemistryRenderPanel {
        @Override
        public void setSelected(boolean isSelected) {
            // dont do anything..
        }
    }

}