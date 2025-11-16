"""
用户反馈处理模块

该模块提供用户操作反馈功能，包括进度条、状态消息、成功/错误提示等。
用于改善用户体验，让用户了解操作进度和结果。
"""

import streamlit as st
from typing import Optional, Callable, Any, List
from contextlib import contextmanager
import time


class FeedbackHandler:
    """
    用户反馈处理器
    
    提供统一的用户反馈接口，包括进度条、状态消息、成功/错误提示等。
    """
    
    @staticmethod
    def show_success(message: str, icon: str = "✅") -> None:
        """
        显示成功消息
        
        Args:
            message: 成功消息内容
            icon: 图标（默认为✅）
        """
        st.success(f"{icon} {message}")
    
    @staticmethod
    def show_error(message: str, icon: str = "❌") -> None:
        """
        显示错误消息
        
        Args:
            message: 错误消息内容
            icon: 图标（默认为❌）
        """
        st.error(f"{icon} {message}")
    
    @staticmethod
    def show_warning(message: str, icon: str = "⚠️") -> None:
        """
        显示警告消息
        
        Args:
            message: 警告消息内容
            icon: 图标（默认为⚠️）
        """
        st.warning(f"{icon} {message}")
    
    @staticmethod
    def show_info(message: str, icon: str = "ℹ️") -> None:
        """
        显示信息消息
        
        Args:
            message: 信息消息内容
            icon: 图标（默认为ℹ️）
        """
        st.info(f"{icon} {message}")
    
    @staticmethod
    @contextmanager
    def show_spinner(message: str = "处理中..."):
        """
        显示加载旋转器（上下文管理器）
        
        Args:
            message: 加载消息
            
        使用示例:
            with FeedbackHandler.show_spinner("正在加载数据..."):
                # 执行耗时操作
                load_data()
        """
        with st.spinner(message):
            yield
    
    @staticmethod
    def show_progress_bar(
        items: List[Any],
        process_func: Callable[[Any], Any],
        message_template: str = "处理中... {current}/{total}",
        success_message: Optional[str] = None
    ) -> List[Any]:
        """
        显示进度条并处理项目列表
        
        Args:
            items: 要处理的项目列表
            process_func: 处理单个项目的函数
            message_template: 进度消息模板，支持{current}和{total}占位符
            success_message: 完成后的成功消息（可选）
            
        Returns:
            处理结果列表
            
        使用示例:
            def collect_shell_data(shell_id):
                return collector.collect(shell_id)
            
            results = FeedbackHandler.show_progress_bar(
                shell_ids,
                collect_shell_data,
                "正在收集数据... {current}/{total}",
                "数据收集完成！"
            )
        """
        total = len(items)
        results = []
        
        # 创建进度条
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            for idx, item in enumerate(items):
                # 更新状态文本
                current = idx + 1
                status_msg = message_template.format(current=current, total=total)
                status_text.text(status_msg)
                
                # 处理项目
                result = process_func(item)
                results.append(result)
                
                # 更新进度条
                progress = current / total
                progress_bar.progress(progress)
                
                # 短暂延迟，让用户看到进度更新
                time.sleep(0.05)
            
            # 清除进度显示
            progress_bar.empty()
            status_text.empty()
            
            # 显示成功消息
            if success_message:
                FeedbackHandler.show_success(success_message)
            
            return results
            
        except Exception as e:
            # 清除进度显示
            progress_bar.empty()
            status_text.empty()
            
            # 显示错误
            FeedbackHandler.show_error(f"处理过程中出错: {str(e)}")
            raise
    
    @staticmethod
    def show_collection_progress(
        shell_ids: List[str],
        data_sources: dict,
        collect_func: Callable[[str, dict], dict]
    ) -> dict:
        """
        显示数据收集进度
        
        专门用于数据收集操作的进度显示。
        
        Args:
            shell_ids: 壳体号列表
            data_sources: 数据源配置字典
            collect_func: 收集函数，接收(shell_id, data_sources)，返回收集结果
            
        Returns:
            收集结果字典
        """
        total_shells = len(shell_ids)
        
        # 计算总步骤数（每个壳体 × 启用的数据源数量）
        enabled_sources = sum(1 for v in data_sources.values() if v)
        total_steps = total_shells * enabled_sources
        
        # 创建进度显示
        progress_bar = st.progress(0)
        status_text = st.empty()
        detail_text = st.empty()
        
        results = {}
        current_step = 0
        
        try:
            for idx, shell_id in enumerate(shell_ids):
                # 更新主状态
                status_text.text(f"📦 正在收集壳体 {idx + 1}/{total_shells}: {shell_id}")
                
                # 显示数据源详情
                source_details = []
                if data_sources.get('data_fetch'):
                    source_details.append("Data_fetch")
                if data_sources.get('test_analysis'):
                    source_details.append("TestAnalysis")
                if data_sources.get('progress'):
                    source_details.append("Progress")
                
                detail_text.text(f"   数据源: {', '.join(source_details)}")
                
                # 收集数据
                result = collect_func(shell_id, data_sources)
                results[shell_id] = result
                
                # 更新进度
                current_step += enabled_sources
                progress = current_step / total_steps
                progress_bar.progress(min(progress, 1.0))
                
                # 短暂延迟
                time.sleep(0.05)
            
            # 清除进度显示
            progress_bar.empty()
            status_text.empty()
            detail_text.empty()
            
            # 显示成功消息
            FeedbackHandler.show_success(
                f"数据收集完成！共收集 {total_shells} 个壳体的数据"
            )
            
            return results
            
        except Exception as e:
            # 清除进度显示
            progress_bar.empty()
            status_text.empty()
            detail_text.empty()
            
            # 显示错误
            FeedbackHandler.show_error(f"数据收集过程中出错: {str(e)}")
            raise
    
    @staticmethod
    def show_loading_status(message: str = "正在加载...") -> Any:
        """
        显示加载状态占位符
        
        Args:
            message: 加载消息
            
        Returns:
            状态占位符对象，可用于后续更新
            
        使用示例:
            status = FeedbackHandler.show_loading_status("正在加载数据...")
            # 执行加载操作
            data = load_data()
            status.empty()  # 清除加载状态
        """
        return st.empty().info(f"⏳ {message}")
    
    @staticmethod
    def show_operation_result(
        success: bool,
        success_message: str,
        error_message: str,
        details: Optional[str] = None
    ) -> None:
        """
        显示操作结果
        
        Args:
            success: 操作是否成功
            success_message: 成功消息
            error_message: 错误消息
            details: 详细信息（可选）
        """
        if success:
            FeedbackHandler.show_success(success_message)
            if details:
                with st.expander("查看详情"):
                    st.text(details)
        else:
            FeedbackHandler.show_error(error_message)
            if details:
                with st.expander("查看错误详情"):
                    st.text(details)
    
    @staticmethod
    def show_validation_feedback(
        is_valid: bool,
        messages: List[str],
        title: str = "验证结果"
    ) -> None:
        """
        显示验证反馈
        
        Args:
            is_valid: 验证是否通过
            messages: 验证消息列表（错误或警告）
            title: 标题
        """
        if is_valid:
            if messages:
                # 有警告但验证通过
                FeedbackHandler.show_warning(f"{title}: 通过（有 {len(messages)} 个警告）")
                with st.expander("查看警告详情"):
                    for msg in messages:
                        st.warning(f"⚠️ {msg}")
            else:
                # 完全通过
                FeedbackHandler.show_success(f"{title}: 通过")
        else:
            # 验证失败
            FeedbackHandler.show_error(f"{title}: 失败（{len(messages)} 个错误）")
            with st.expander("查看错误详情", expanded=True):
                for msg in messages:
                    st.error(f"❌ {msg}")
    
    @staticmethod
    def show_detailed_validation_feedback(
        is_valid: bool,
        messages: List[str],
        title: str = "数据验证",
        show_suggestions: bool = True
    ) -> None:
        """
        显示详细的验证反馈，包括分类和修复建议
        
        Args:
            is_valid: 验证是否通过
            messages: 验证消息列表
            title: 标题
            show_suggestions: 是否显示修复建议
        """
        if is_valid:
            if messages:
                # 有警告但验证通过
                st.warning(f"⚠️ {title}: 通过（有 {len(messages)} 个警告）")
                
                # 分类警告
                field_warnings = [m for m in messages if '缺少' in m or '字段' in m]
                data_warnings = [m for m in messages if '超出' in m or '范围' in m]
                other_warnings = [m for m in messages if m not in field_warnings and m not in data_warnings]
                
                with st.expander("查看警告详情", expanded=False):
                    if field_warnings:
                        st.markdown("**🟡 字段警告:**")
                        for msg in field_warnings:
                            st.caption(f"  • {msg}")
                    
                    if data_warnings:
                        st.markdown("**🟠 数据警告:**")
                        for msg in data_warnings:
                            st.caption(f"  • {msg}")
                    
                    if other_warnings:
                        st.markdown("**⚠️ 其他警告:**")
                        for msg in other_warnings:
                            st.caption(f"  • {msg}")
                    
                    if show_suggestions:
                        st.divider()
                        st.markdown("**💡 建议:**")
                        st.caption("  • 这些警告不会影响数据的使用")
                        st.caption("  • 建议在下次数据收集时补充缺失的字段")
            else:
                # 完全通过
                st.success(f"✅ {title}: 完全通过，数据完整且格式正确")
        else:
            # 验证失败
            st.error(f"❌ {title}: 失败（{len(messages)} 个错误）")
            
            # 分类错误
            critical_errors = [m for m in messages if '缺少' in m and ('metadata' in m or 'shells' in m)]
            field_errors = [m for m in messages if '字段' in m or '类型' in m]
            data_errors = [m for m in messages if m not in critical_errors and m not in field_errors]
            
            with st.expander("查看错误详情", expanded=True):
                if critical_errors:
                    st.markdown("**🔴 关键错误:**")
                    for msg in critical_errors:
                        st.error(f"  • {msg}")
                
                if field_errors:
                    st.markdown("**🟡 字段错误:**")
                    for msg in field_errors:
                        st.error(f"  • {msg}")
                
                if data_errors:
                    st.markdown("**🟠 数据错误:**")
                    for msg in data_errors:
                        st.error(f"  • {msg}")
                
                if show_suggestions:
                    st.divider()
                    st.markdown("**💡 修复建议:**")
                    if critical_errors:
                        st.caption("  • 检查文件是否为有效的数据集格式")
                        st.caption("  • 确保包含必需的 metadata 和 shells 字段")
                    if field_errors:
                        st.caption("  • 检查数据字段类型是否正确")
                        st.caption("  • 确保所有必填字段都已填写")
                    if data_errors:
                        st.caption("  • 检查数据值是否在合理范围内")
                        st.caption("  • 验证数值数据是否为有效数字")
    
    @staticmethod
    def confirm_action(
        message: str,
        confirm_text: str = "确认",
        cancel_text: str = "取消"
    ) -> bool:
        """
        显示确认对话框
        
        Args:
            message: 确认消息
            confirm_text: 确认按钮文本
            cancel_text: 取消按钮文本
            
        Returns:
            用户是否确认
            
        注意: 这个函数需要配合session_state使用
        """
        st.warning(message)
        col1, col2 = st.columns(2)
        
        with col1:
            confirmed = st.button(confirm_text, type="primary", use_container_width=True)
        with col2:
            cancelled = st.button(cancel_text, use_container_width=True)
        
        return confirmed and not cancelled
