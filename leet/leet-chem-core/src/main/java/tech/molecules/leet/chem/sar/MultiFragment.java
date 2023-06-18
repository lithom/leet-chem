package tech.molecules.leet.chem.sar;

import com.actelion.research.chem.StereoMolecule;

import java.util.ArrayList;
import java.util.List;

public class MultiFragment {
    private List<SARElement> elements;

    public MultiFragment(List<SARElement> elements) {
        this.elements = elements;
    }

    public MultiFragment() {
        SARElement ea = new SARElement(new StereoMolecule());
        List<SARElement> sel = new ArrayList<>();
        sel.add(ea);
        this.elements = sel;
    }

    public List<SARElement> getElements() {
        return elements;
    }

    private List<MultiFragmentListener> elementListeners = new ArrayList<>();

    public void addMultiFragmentElement(SARElement element) {
        elements.add(element);
        fireMultiFragmentElementAdded(element);
    }

    public void removeMultiFragmentElement(int index) {
        elements.remove(index);
        fireMultiFragmentElementRemoved(index);
    }

    public void setFragmentEdited(SARElement element) {
        this.fireSARElementSelected(element);
    }

    public void addMultiFragmentElementListener(MultiFragmentListener listener) {
        elementListeners.add(listener);
    }

    public void removeMultiFragmentElementListener(MultiFragmentListener listener) {
        elementListeners.remove(listener);
    }

    private void fireMultiFragmentElementAdded(SARElement element) {
        for (MultiFragmentListener listener : elementListeners) {
            listener.onMultiFragmentElementAdded(element);
        }
    }

    private void fireMultiFragmentElementRemoved(int index) {
        for (MultiFragmentListener listener : elementListeners) {
            listener.onMultiFragmentElementRemoved(index);
        }
    }

    private void fireSARElementSelected(SARElement element) {
        for (MultiFragmentListener listener : elementListeners) {
            listener.onSARElementSelected(element);
        }
    }

    public interface MultiFragmentListener {
        void onMultiFragmentElementAdded(SARElement element);
        void onMultiFragmentElementRemoved(int index);
        void onSARElementSelected(SARElement element);
    }

}
