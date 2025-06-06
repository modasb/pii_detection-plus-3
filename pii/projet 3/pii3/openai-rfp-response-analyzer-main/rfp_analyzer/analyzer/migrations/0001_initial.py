# Generated by Django 5.0.3 on 2025-03-18 01:31

import django.db.models.deletion
import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Analysis',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(default=django.utils.timezone.now)),
                ('report', models.JSONField(default=dict)),
            ],
        ),
        migrations.CreateModel(
            name='Document',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=255)),
                ('document_type', models.CharField(choices=[('RFP', 'Request for Proposal'), ('RESPONSE', 'Response')], max_length=10)),
                ('file', models.FileField(upload_to='documents/')),
                ('uploaded_at', models.DateTimeField(default=django.utils.timezone.now)),
                ('processed', models.BooleanField(default=False)),
                ('faiss_index_path', models.CharField(blank=True, max_length=255)),
            ],
        ),
        migrations.CreateModel(
            name='ChatMessage',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('question', models.TextField()),
                ('answer', models.TextField()),
                ('created_at', models.DateTimeField(default=django.utils.timezone.now)),
                ('analysis', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='chat_messages', to='analyzer.analysis')),
            ],
        ),
        migrations.AddField(
            model_name='analysis',
            name='response_document',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='response_analyses', to='analyzer.document'),
        ),
        migrations.AddField(
            model_name='analysis',
            name='rfp_document',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='rfp_analyses', to='analyzer.document'),
        ),
    ]
