# Refactor R2A — Tạo `BasicCalculatorController`

Sau R1B, phần Scientific đã được chuyển khỏi `MainActivity` sang:

```text
ScientificCalculatorController
```

Bước tiếp theo là chuẩn bị tách phần Basic Calculator.

Ở R2A:

```text
Tạo 1 file mới
Không sửa MainActivity
Không xóa Basic code cũ
Không thay đổi hành vi app
```

Controller mới chưa được nối vào `MainActivity`. App vẫn chạy bằng Basic implementation hiện tại trong Activity.

---

## 1. Tạo file

Đường dẫn:

```text
app/src/main/java/com/example/calculator/ui/controller/
BasicCalculatorController.java
```

Package:

```java
package com.example.calculator.ui.controller;
```

Thay toàn bộ file bằng:

```java
package com.example.calculator.ui.controller;

import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import com.example.calculator.R;
import com.example.calculator.engine.CalculatorEngine;
import com.example.calculator.engine.CalculatorState;
import com.example.calculator.history.HistoryManager;

/**
 * Điều phối toàn bộ giao diện Basic Calculator.
 *
 * Trách nhiệm:
 *
 * - Bind Basic Calculator Views
 * - Nhận button click
 * - Gửi action sang CalculatorEngine
 * - Đọc CalculatorState
 * - Render expression/result
 * - Lưu History sau phép tính hợp lệ
 *
 * Không thực hiện phép tính trực tiếp.
 */
public final class BasicCalculatorController {

    // =========================================================
    // DEPENDENCIES
    // =========================================================

    private final View rootView;

    private final CalculatorEngine calculatorEngine;

    private final HistoryManager historyManager;


    // =========================================================
    // DISPLAY VIEWS
    // =========================================================

    private TextView tvExpression;

    private TextView tvResult;


    // =========================================================
    // CONSTRUCTOR
    // =========================================================

    public BasicCalculatorController(
            View rootView,
            CalculatorEngine calculatorEngine,
            HistoryManager historyManager
    ) {

        if (rootView == null) {

            throw new IllegalArgumentException(
                    "Basic root view cannot be null."
            );
        }


        if (calculatorEngine == null) {

            throw new IllegalArgumentException(
                    "CalculatorEngine cannot be null."
            );
        }


        this.rootView =
                rootView;


        this.calculatorEngine =
                calculatorEngine;


        this.historyManager =
                historyManager;
    }


    // =========================================================
    // PUBLIC API
    // =========================================================

    /**
     * Bind Views, setup listeners và render state ban đầu.
     *
     * Chỉ gọi một lần sau khi activity_main.xml được inflate.
     */
    public void setup() {

        bindViews();

        setupDigitButtons();

        setupOperatorButtons();

        setupFunctionButtons();

        renderState();
    }


    /**
     * Đưa Basic Calculator về trạng thái ban đầu.
     */
    public void clear() {

        calculatorEngine.clear();

        renderState();
    }


    // =========================================================
    // BIND VIEWS
    // =========================================================

    private void bindViews() {

        tvExpression =
                findBasicView(
                        R.id.tvExpression
                );


        tvResult =
                findBasicView(
                        R.id.tvResult
                );
    }


    private <T extends View> T findBasicView(
            int viewId
    ) {

        return rootView.findViewById(
                viewId
        );
    }


    // =========================================================
    // DIGIT BUTTONS
    // =========================================================

    private void setupDigitButtons() {

        bindDigitButton(
                R.id.btn0,
                "0"
        );

        bindDigitButton(
                R.id.btn1,
                "1"
        );

        bindDigitButton(
                R.id.btn2,
                "2"
        );

        bindDigitButton(
                R.id.btn3,
                "3"
        );

        bindDigitButton(
                R.id.btn4,
                "4"
        );

        bindDigitButton(
                R.id.btn5,
                "5"
        );

        bindDigitButton(
                R.id.btn6,
                "6"
        );

        bindDigitButton(
                R.id.btn7,
                "7"
        );

        bindDigitButton(
                R.id.btn8,
                "8"
        );

        bindDigitButton(
                R.id.btn9,
                "9"
        );
    }


    private void bindDigitButton(
            int buttonId,
            String digit
    ) {

        Button button =
                findBasicView(
                        buttonId
                );


        button.setOnClickListener(view -> {

            calculatorEngine.inputDigit(
                    digit
            );


            renderState();
        });
    }


    // =========================================================
    // OPERATOR BUTTONS
    // =========================================================

    private void setupOperatorButtons() {

        bindOperatorButton(
                R.id.btnAdd,
                "+"
        );

        bindOperatorButton(
                R.id.btnSubtract,
                "−"
        );

        bindOperatorButton(
                R.id.btnMultiply,
                "×"
        );

        bindOperatorButton(
                R.id.btnDivide,
                "÷"
        );
    }


    private void bindOperatorButton(
            int buttonId,
            String operator
    ) {

        Button button =
                findBasicView(
                        buttonId
                );


        button.setOnClickListener(view -> {

            calculatorEngine.inputOperator(
                    operator
            );


            renderState();
        });
    }


    // =========================================================
    // FUNCTION BUTTONS
    // =========================================================

    private void setupFunctionButtons() {

        // Decimal
        findBasicView(
                R.id.btnDecimal
        ).setOnClickListener(view -> {

            calculatorEngine.inputDecimal();

            renderState();
        });


        // Equals
        findBasicView(
                R.id.btnEquals
        ).setOnClickListener(view -> {

            calculateAndSaveHistory();
        });


        // AC
        findBasicView(
                R.id.btnClear
        ).setOnClickListener(view -> {

            clear();
        });


        // Backspace
        findBasicView(
                R.id.btnBackspace
        ).setOnClickListener(view -> {

            calculatorEngine.backspace();

            renderState();
        });


        // Toggle sign
        findBasicView(
                R.id.btnToggleSign
        ).setOnClickListener(view -> {

            calculatorEngine.toggleSign();

            renderState();
        });


        // Percent
        findBasicView(
                R.id.btnPercent
        ).setOnClickListener(view -> {

            calculatorEngine.percent();

            renderState();
        });
    }


    // =========================================================
    // CALCULATE + HISTORY
    // =========================================================

    /**
     * Thực hiện calculation và chỉ lưu History
     * nếu "=" thực sự hoàn thành một phép tính hợp lệ.
     *
     * Không lưu:
     *
     * - Error
     * - "=" khi chưa có phép tính
     * - "=" lặp lại nhiều lần
     */
    private void calculateAndSaveHistory() {

        CalculatorState beforeState =
                calculatorEngine.getState();


        boolean hasPendingCalculation =

                !beforeState.isError()

                        && beforeState.getFirstOperand() != null

                        && beforeState.getOperator() != null

                        && !beforeState.isWaitingForOperand();


        calculatorEngine.calculate();


        CalculatorState afterState =
                calculatorEngine.getState();


        if (hasPendingCalculation

                && !afterState.isError()

                && afterState.getExpression() != null

                && !afterState
                        .getExpression()
                        .trim()
                        .isEmpty()

                && afterState.getResult() != null

                && !afterState
                        .getResult()
                        .trim()
                        .isEmpty()) {


            saveHistory(

                    afterState.getExpression(),

                    afterState.getResult()
            );
        }


        renderState();
    }


    private void saveHistory(
            String expression,
            String result
    ) {

        if (historyManager == null) {

            return;
        }


        if (expression == null
                || expression.trim().isEmpty()) {

            return;
        }


        if (result == null
                || result.trim().isEmpty()
                || "Error".equals(result)) {

            return;
        }


        historyManager.add(

                expression,

                result
        );
    }


    // =========================================================
    // RENDER
    // =========================================================

    private void renderState() {

        CalculatorState state =
                calculatorEngine.getState();


        String expression =
                state.getExpression();


        String result =
                state.getResult();


        tvExpression.setText(

                expression == null

                        ? ""
                        : expression
        );


        tvResult.setText(

                result == null

                        ? "0"
                        : result
        );
    }
}
```

---

## 2. Không sửa `MainActivity` ở R2A

Giữ nguyên `MainActivity.java` từ R1B.

Chưa xóa các field Basic:

```text
CalculatorEngine calculatorEngine

TextView tvExpression
TextView tvResult
```

Chưa xóa các method Basic:

```text
setupDigitButtons()
bindDigitButton()

setupOperatorButtons()
bindOperatorButton()

setupFunctionButtons()

calculateAndSaveHistory()

renderState()
```

Các field và method này vẫn là implementation thật cho đến R2B.

Controller mới ở R2A chỉ cần compile độc lập.

---

## 3. Build checkpoint

Chạy:

```text
Build
→ Make Project
```

Checklist:

```text
[ ] BasicCalculatorController.java compile

[ ] Package ui.controller resolve

[ ] CalculatorEngine import resolve

[ ] CalculatorState import resolve

[ ] HistoryManager import resolve

[ ] Không báo thiếu ID:
    tvExpression
    tvResult
    btn0–btn9
    btnAdd
    btnSubtract
    btnMultiply
    btnDivide
    btnDecimal
    btnEquals
    btnClear
    btnBackspace
    btnToggleSign
    btnPercent

[ ] MainActivity chưa thay đổi

[ ] App vẫn chạy như checkpoint R1B
```

---

## 4. Kiểm tra nhanh sau Build

Do controller chưa được nối, behavior app phải giữ nguyên:

```text
Basic hoạt động
Scientific hoạt động
History hoạt động
Mode switching hoạt động
Convert placeholder hiển thị
```

Không cần test chi tiết controller ở R2A vì chưa có listener nào từ file mới đang được sử dụng.

---

## 5. Trạng thái refactor

```text
[✓] R1A
    Tạo ScientificCalculatorController

[~] R1B
    Nối Scientific controller vào MainActivity
    Chờ xác nhận Build/Run

[~] R2A
    Tạo BasicCalculatorController
    Chờ Build

NEXT

R2B
→ nối BasicCalculatorController vào MainActivity
→ xóa toàn bộ Basic fields/methods khỏi MainActivity
→ Build
→ regression test Basic + Scientific + History
```
